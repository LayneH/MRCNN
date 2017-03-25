import numpy as np
import os
from argparse import ArgumentParser
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from cifar_data import *
from mani import *
from vgg import load_vgg_weights, extract_vgg_feature

types = ["fruit_and_vegetables", "household_electrical_devices"]
VGG_PATH = 'vgg/imagenet-vgg-verydeep-19.mat'
RESULT_PATH = 'result.csv'

ALPHA = 1e-4
BETA = 1e-2
K = 3
LEARNING_RATE = 10**2.5
ITERATION = 200
VERBOSE = 50

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--vgg_path', dest='vgg', metavar='VGG_PATH',
        help='directory of pretrained VGG Net', default=VGG_PATH)
    parser.add_argument('--result_path', dest='out_file', metavar='RESULT_PATH',
        help='file that save the result', default=RESULT_PATH)
    parser.add_argument('--gpu_id', dest='gpu_id', metavar='GPU_ID',
        help='select GPU devices to use', default=None)
    return parser

def load_vgg_feats(vgg_W, mean_im, feats_path='vgg_feats.npy'):
    feats = {}
    if !os.isfile(feats_path):
        data = load_data()
        for super_c in data:
            feats[super_c] = {}
            for c in data[super_c]:
                feats[super_c][c] = extract_vgg_feature(data[super_c][c] - mean_im, vgg_W)
        with open(feats_path, 'wb') as f:
            lap = np.save(f)
    else:
        with open(feats_path, 'rb') as f:
            feats = np.load(f).item()
    return feats

def load_lap(data, lap_path='lap.npy'):
    lap = {}
    if !os.isfile(lap_path):
        for i, pos in data[types[0]].iteritems():
            for j, neg in data[types[1]].iteritems():
                M = get_kNN(np.concatenate((pos, neg), axis=0), k)
                lap['%s-%s'%(i, j)] = np.diag(np.sum(M, axis=1)) - M
        with open(lap_path, 'wb') as f:
            lap = np.save(f)
    else:
        with open(lap_path, 'rb') as f:
            lap = np.load(f).item()
    return lap

def get_problems():
    problems = []
    for pos1 in data[types[0]]:
        for pos2 in data[types[0]]:
            if pos1 == pos2:
                continue
            for neg1 in data[types[1]]:
                for neg2 in data[types[1]]:
                    if neg1 == neg2:
                        continue
                    problem = "%s-%s-vs-%s-%s"%(pos1, neg1, pos2, neg2)
                    problems.append(problem)
    return problems

def train_and_test(src, targ, Ws, Lap, param={}):
    '''
    src - source domain data
    targ - target domain data
    '''
    # get keywords
    alpha = param.pop('alpha', 0.1)
    beta = param.pop('beta', 1)
    lr = param.pop('lr', 1e-5)
    max_steps = param.pop('steps', 500)
    epochs_per_decay = param.pop('epochs_per_decay', 10)
    verbose = param.pop('verbose', 0)
    k = param.pop('k', 10)

    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        x_targ = tf.placeholder(tf.float32, [None, 2, 2, 512])
        x_src = tf.placeholder(tf.float32, [None, 2, 2, 512])
        y_src = tf.placeholder(tf.float32, [None, 2])

        # build the model
        net = vgg_upper(Ws)
        xe, reg_penalty, mani, acc_op = net.vgg_trans(x_src, x_targ, y_src, Lap)

        # overall loss
        loss_op = tf.multiply(alpha, mani) + tf.multiply(beta, xe) + tf.multiply(gamma, reg_penalty)

        # decay the learning rate
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(lr, global_step*batch_size,
                                               epochs_per_decay*src.size, 0.5,staircase=True)

        # here we apply smaller learning rate on conv5 layers
        var_list1 = tf.trainable_variables()[:8]
        var_list2 = tf.trainable_variables()[8:]
        opt1 = tf.train.AdamOptimizer(learning_rate / 100.)
        opt2 = tf.train.AdamOptimizer(learning_rate)
        grads = tf.gradients(loss_op, var_list1 + var_list2)
        grads1 = grads[:len(var_list1)]
        grads2 = grads[len(var_list1):]
        train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
        train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
        train_step = tf.group(train_op1, train_op2)

    # start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #
    # config.gpu_options.per_process_gpu_memory_fraction = 0
    config.allow_soft_placement=True
    with tf.Session(graph=g, config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in xrange(max_steps):
            train_loss = sess.run([train_step, loss_op], feed_dict={x_src:src.images,
                y_src:src.labels, x_targ:targ.images})
            if verbose > 0 and ((i+1)%verbose == 0 or i+1 == max_steps or i == 0):
                print '%4d/%4d loss %f'%(i+1,
                        max_steps, train_loss, train_acc, targ_loss, targ_acc, mnf)
        targ_acc = sess.run(acc_op, feed_dict={x_src:targ.images, y_src:targ.labels})
    return targ_acc

def main():
    parser = get_parser()
    option = parser.parse_args()

    assert os.isfile(option.vgg_path), "You may forget to download pretrained\
                                        VGG Net, or please specify the model path\
                                        by --vgg_path option if you have the\
                                        model already."
    if option.gpu_id is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    vgg_W, mean_im = load_vgg_weights()
    feats = load_vgg_feats(vgg_W, mean_im)
    lap = load_lap(feats)
    problems = get_problems()
    acc = np.zeros((len(problems), ))
    param={'lr':LEARNING_RATE, 'alpha':ALPHA, 'beta':BETA, 'k':K,
        'steps':ITERATION, 'verbose':VERBOSE, 'epochs_per_decay':20}
    for i, problem in enumerate(problems):
        pos1, neg1, _, pos2, neg2 = problem.split('-')
        X_s, y_s = get_XY(feat[types[0]][pos1], feat[types[1]][neg1])
        X_t, y_t = get_XY(feat[types[0]][pos2], feat[types[1]][neg2])
        src = data_holder(X_s, y_s)
        targ = data_holder(X_t, y_t)
        acc[i] = train_and_test(src, targ, param)
        print '%d / %d, %s, accuracy is'%(i+1, len(problem), problem, acc[i])
        with open(option.out_file, 'wb') as f:
            np.savetxt(f, acc*100, delimiter=',', fmt='%.4f')
    duration = end - start
    print 'Average accuracy is ', np.mean(acc)
    print '%d problems take %fs'%(len(problems), duration)

if __name__ == '__main__':
    main()
