import numpy as np
import tensorflow as tf
import scipy.io as sio
import os
from cifar_data import *
from mani import model

def load_weights_and_feats(types, gpu_id):
    vgg_W, mean_im = load_vgg_weights()
    feats = load_vgg_feats(vgg_W, mean_im, types, gpu_id=gpu_id)
    Ws = {}
    for i in xrange(1, 5):
        layer = 'conv5_%d'%i
        Ws[layer] = vgg_W[layer]
    return Ws, feats

def load_vgg_feats(vgg_W, mean_im, types, feats_path='datasets/vgg_feats.npy', gpu_id=None):
    feats = {}
    #print feats_path
    if os.path.isfile(feats_path) == False:
        data = load_data(types)
        feats = extract_vgg_feats(data, mean_im, vgg_W, gpu_id)
        with open(feats_path, 'wb') as f:
            np.save(f, feats)
    else:
        with open(feats_path, 'rb') as f:
            feats = np.load(f).item()
    return feats

def extract_vgg_feats(data, mean_im, Ws, gpu_id=None):
    proc_ims = {}
    if gpu_id is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        net = model(Ws)
        extract_op = net.build_lower(X)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        for super_name, super_class in data.iteritems():
            proc_ims[super_name] = {}
            for sub_name, ims in super_class.iteritems():
                print 'Extracting', sub_name
                proc_ims[super_name][sub_name] = sess.run(extract_op, feed_dict={X:ims-mean_im})
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
