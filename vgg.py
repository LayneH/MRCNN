import numpy as np
import tensorflow as tf
import scipy.io as sio
import os

def extract_vgg_feature(ims, Ws, gpu_id=None):
    if gpu_id is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        h = conv_relu(Ws, X, 'conv1_1')
        h = conv_relu(Ws, h, 'conv1_2')
        h = max_pool(h, 'pool1')

        h = conv_relu(Ws, h, 'conv2_1')
        h = conv_relu(Ws, h, 'conv2_2')
        h = max_pool(h, 'pool2')

        h = conv_relu(Ws, h, 'conv3_1')
        h = conv_relu(Ws, h, 'conv3_2')
        h = conv_relu(Ws, h, 'conv3_3')
        h = conv_relu(Ws, h, 'conv3_4')
        h = max_pool(h, 'pool3')

        h = conv_relu(Ws, h, 'conv4_1')
        h = conv_relu(Ws, h, 'conv4_2')
        h = conv_relu(Ws, h, 'conv4_3')
        h = conv_relu(Ws, h, 'conv4_4')
        h = max_pool(h, 'pool4')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            proc_ims = sess.run(h, feed_dict={X:ims})
    return proc_ims

def load_vgg_weights(fname='vgg/imagenet-vgg-verydeep-19.mat'):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'

        'fc6', 'relu6',  'fc7', 'relu7', 'fc8'
    )

    W = sio.loadmat(fname)
    mean_im = np.mean(W['normalization'][0][0][0], axis=(0, 1))
    weights = W['layers'][0]

    vgg_W = {}
    for i, layer in enumerate(layers):
        if layer[:4] == 'relu' or layer[:4] == 'pool':
            continue
        temp = {}
        temp['W'], temp['b'] = weights[i][0][0][0][0]
        temp['W'] = np.transpose(temp['W'], (1, 0, 2, 3))
        vgg_W[layer] = temp
    return vgg_W, mean_im

def conv_relu(weights, bottom, name, shape=None):
    with tf.name_scope(name):
        kernel, bias = get_var(weights, name)
        conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, bias)
        return tf.nn.relu(conv)

def conv(weights, bottom, name, shape=None):
    with tf.name_scope(name):
        kernel, bias = get_var(weights, name)
        conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, bias)
        return conv

def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def get_var(weights, name):
    k1, k2 = weights[name]['W'], weights[name]['b'].reshape((-1, ))
    W = tf.Variable(k1, name=name+'_W')
    b = tf.Variable(k2, name=name+'_b')
    return W, b
