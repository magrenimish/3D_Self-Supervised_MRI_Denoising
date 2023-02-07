# self-supervised 3d image denoising 
import tensorflow as tf
import numpy as np

import cv2
import os

import util
from helper import Punet as Pnet

TF_DATA_TYPE = tf.float32
LEARNING_RATE = 1e-4
N_PREDICTION = 1 #100
N_SAVE = 1000 #1000
N_STEP = 50000 #150000


def partial_conv_unet(noisy_image, mask, channel=3, depth=256, width=256, height=256, p=0.7, **_kwargs):
        """
        Modified Unet (encoder-decoder) based model architecture with residual connections, dropout and partial convolutions. Partial Convolutions
        only perform the convolution step on the unmasked pixels
        Args:
            noisy_image (tensor): noisy 3d image input
            mask (tensor): bernoulli droput mask; same shape as noisy_image
            channel (int): number of image color channels
            depth (int): image depth
            widht (int): image width
            height (int): image height
            p (float): decoder layers dropout probability
        Returns:
            updated_image (tensor): denoised model output
        """
    noisy_image.set_shape([None, channel, depth, height, width])
    mask.set_shape([None, channel, depth, height, width])
    skips = [noisy_image]
    
    # Encoder layers with appended skip connections 
    updated_image = initial_image
    updated_image, mask = Pnet.Pconv_lr('enc_conv0', updated_image, 48, mask_in=mask)
    updated_image, mask = Pnet.Pconv_lr('enc_conv1', updated_image, 48, mask_in=mask)
    updated_image, mask = Pnet.Pmaxpool3d(updated_image, mask_in=mask)
    skips.append(updated_image)

    updated_image, mask = Pnet.Pconv_lr('enc_conv2', updated_image, 48, mask_in=mask)
    updated_image, mask = Pnet.Pmaxpool3d(updated_image, mask_in=mask)
    skips.append(updated_image)

    updated_image, mask = Pnet.Pconv_lr('enc_conv3', updated_image, 48, mask_in=mask)
    updated_image, mask = Pnet.Pmaxpool3d(updated_image, mask_in=mask)
    skips.append(updated_image)

    updated_image, mask = Pnet.Pconv_lr('enc_conv4', updated_image, 48, mask_in=mask)
    updated_image, mask = Pnet.Pmaxpool3d(updated_image, mask_in=mask)
    skips.append(updated_image)

    updated_image, mask = Pnet.Pconv_lr('enc_conv5', updated_image, 48, mask_in=mask)
    updated_image, mask = Pnet.Pmaxpool3d(updated_image, mask_in=mask)
    updated_image, mask = Pnet.Pconv_lr('enc_conv6', updated_image, 48, mask_in=mask)

    # Decoder layers with added skip connections
    updated_image= Pnet.upscale3d(updated_image)
    updated_image= Pnet.concat(updated_image, skips.pop())
    updated_image= Pnet.conv_lr('dec_conv5', updated_image, 96, p=p)
    updated_image= Pnet.conv_lr('dec_conv5b', updated_image, 96, p=p)

    updated_image= Pnet.upscale3d(updated_image)
    updated_image = Pnet.concat(updated_image, skips.pop())
    updated_image= Pnet.conv_lr('dec_conv4', updated_image, 96, p=p)
    updated_image= Pnet.conv_lr('dec_conv4b', updated_image, 96, p=p)

    updated_image= Pnet.upscale3d(updated_image)
    updated_image= Pnet.concat(updated_image, skips.pop())
    updated_image= Pnet.conv_lr('dec_conv3', updated_image, 96, p=p)
    updated_image= Pnet.conv_lr('dec_conv3b', updated_image, 96, p=p)

    updated_image= Pnet.upscale3d(updated_image)
    updated_image= Pnet.concat(updated_image, skips.pop())
    updated_image= Pnet.conv_lr('dec_conv2', updated_image, 96, p=p)
    updated_image= Pnet.conv_lr('dec_conv2b', updated_image, 96, p=p)

    updated_image= Pnet.upscale3d(updated_image)
    updated_image= Pnet.concat(updated_image, skips.pop())
    updated_image= Pnet.conv_lr('dec_conv1a', updated_image, 64, p=p)
    updated_image= Pnet.conv_lr('dec_conv1b', updated_image, 32, p=p)
    updated_image= Pnet.conv('dec_conv1', updated_image, channel, p=p)

    return updated_image


def build_denoising_unet(noisy, p=0.7, is_realnoisy=True):
        """
        Modified Unet model definition 
        Args:
            noisy (numpy array): input 3d noisy image
            p (float): decoder layers dropout probability
            is_realnoisy (bool): true if image ground truth is noisy
        Returns:
            model (object): tensorflow nn module
        """
    _, d, h, w, c = np.shape(noisy)
    noisy_tensor = tf.identity(noisy)
    response = tf.transpose(noisy_tensor, [0, 4, 1, 2, 3])
    mask_tensor = tf.ones_like(response)
    mask_tensor = tf.nn.dropout(mask_tensor, 0.7) * 0.7
    # response is the output of bernoulli drouput used on the intial image
    response = tf.multiply(mask_tensor, response)
    slice_avg = tf.get_variable('slice_avg', shape=[_, d, h, w, c], initializer=tf.initializers.zeros())
    response = partial_conv_unet(response, mask_tensor, channel=c, depth=d, width=w, height=h, p=p)
    response = tf.transpose(response, [0, 2, 3, 4, 1])
    mask_tensor = tf.transpose(mask_tensor, [0, 2, 3, 4, 1])
    data_loss = Pnet.mask_loss(response, noisy_tensor, 1. - mask_tensor)
    avg_op = slice_avg.assign(slice_avg * 0.99 + response * 0.01)
    out_image = response

    training_error = data_loss
    tf.summary.scalar('data loss', data_loss)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=3)
    model = {
        'training_error': training_error,
        'data_loss': data_loss,
        'saver': saver,
        'summary': merged,
        'out_image': out_image,
        'avg_op': avg_op,
        'slice_avg': slice_avg,
    }

    return model


def train(file_path, dropout_rate, sigma=25, is_realnoisy=True):
        """
        function to train the model 
        Args:
            file_path (string): path to noisy input image
            drouput_rate (float): decoder layers dropout probability
            sigma (int): standard deviation to add gaussian noise in the absence of real noise
            is_realnoisy (bool): true if image ground truth is noisy
        """    
    print(file_path)
    tf.reset_default_graph()
    gt = util.load_np_image(file_path)
    _, d, w, h, c = np.shape(gt)
    model_path = file_path[0:file_path.rfind(".")] + "/" + str(sigma) + "/model/Slef_spuervised_denoising/"
    os.makedirs(model_path, exist_ok=True)
    noisy = util.add_gaussian_noise(gt, model_path, sigma)
    model = build_denoising_unet(noisy, 1 - dropout_rate, is_realnoisy)

    loss = model['training_error']
    summay = model['summary']
    saver = model['saver']
    out_image = model['out_image']
    avg_op = model['avg_op']
    slice_avg = model['slice_avg']
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    avg_loss = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(model_path, sess.graph)
        for step in range(N_STEP):
            _, _op, loss_value, merged, o_image = sess.run([optimizer, avg_op, loss, summay, out_image])
            avg_loss += loss_value          
            if (step + 1) % N_SAVE == 0:

                print("After %d training step(s)" % (step + 1),
                      "loss  is {:.9f}".format(avg_loss / N_SAVE))
                avg_loss = 0
                sum = np.float32(np.zeros(out_image.shape.as_list()))
                for j in range(N_PREDICTION):
                    o_avg, o_image = sess.run([slice_avg, out_image])
                    sum += o_image
                o_image = np.squeeze(np.uint8(np.clip(sum / N_PREDICTION, 0, 1) * 255))
                o_avg = np.squeeze(np.uint8(np.clip(o_avg, 0, 1) * 255))
                if is_realnoisy:
                    np.save(model_path + 'Self_supervised_denoising-' + str(step + 1) + '.npy', o_avg)
                else:
                    np.save(model_path + 'Self_supervised_denoising-' + str(step + 1) + '.npy', o_image)
                saver.save(sess, model_path + "model.ckpt-" + str(step + 1))
            summary_writer.add_summary(merged, step)


if __name__ == '__main__':
    path = './testsets/MRI/'
    file_list = os.listdir(path)
    sigma = -1
    for file_name in file_list:
        if not os.path.isdir(path + file_name):
            train(path + file_name, 0.3, sigma, is_realnoisy = True)
