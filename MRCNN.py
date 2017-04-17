import numpy as np
import os
import time
from argparse import ArgumentParser
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from vgg import load_weights_and_feats
from mani import load_lap, model
from cifar_data import get_XY, data_holder

types = ["fruit_and_vegetables", "household_electrical_devices"]
VGG_PATH = 'vgg/imagenet-vgg-verydeep-19.mat'
RESULT_PATH = 'result.csv'

ALPHA = 1e-4
BETA = 1e-2
K = 3
LEARNING_RATE = 10**-2.5
ITERATION = 200
VERBOSE = 0
EPOCHS_PER_DECAY = 20

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--vgg_path', dest='vgg_path', metavar='VGG_PATH',
        help='directory of pretrained VGG Net', default=VGG_PATH)
    parser.add_argument('--result_path', dest='out_file', metavar='RESULT_PATH',
        help='file that save the result', default=RESULT_PATH)
    parser.add_argument('--gpu_id', dest='gpu_id', metavar='GPU_ID',
        help='select GPU devices to use', default='0')
    return parser

def get_problems(data):
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
    alpha = param.pop('alpha', ALPHA)
    beta = param.pop('beta', BETA)
    k = param.pop('k', K)
    lr = param.pop('lr', LEARNING_RATE)
    max_steps = param.pop('steps', ITERATION)
    verbose = param.pop('verbose', VERBOSE)
    epochs_per_decay = param.pop('epochs_per_decay', EPOCHS_PER_DECAY)

    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        x_targ = tf.placeholder(tf.float32, [None, 2, 2, 512])
        x_src = tf.placeholder(tf.float32, [None, 2, 2, 512])
        y_src = tf.placeholder(tf.float32, [None, 2])

        # build the model
        net = model(Ws)
        xe, reg_penalty, mani, acc_op = net.build(x_src, x_targ, y_src, Lap)

        # overall loss
        loss_op = xe + tf.multiply(alpha, mani) + tf.multiply(beta, reg_penalty)

        # decay the learning rate
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(lr, global_step,
                            epochs_per_decay, 0.5,staircase=True)

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

    assert os.path.isfile(option.vgg_path), "You may forget to download pretrained "\
                                        "VGG Net, or please specify the model path "\
                                        "by --vgg_path option if you have the "\
                                        "model already."
    if option.gpu_id is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = option.gpu_id
    Ws, feats = load_weights_and_feats(types, option.gpu_id)
    lap = load_lap(feats, K, types)
    problems = get_problems(feats)
    acc = np.zeros((len(problems), ))
    # param={'lr':LEARNING_RATE, 'alpha':ALPHA, 'beta':BETA, 'k':K,
    # 'steps':ITERATION, 'verbose':VERBOSE, 'epochs_per_decay':EPOCHS_PER_DECAY}
    start = time.time()
    last_start = start
    for i, problem in enumerate(problems):
        print '%4d / %4d, %s'%(i+1, len(problem), problem)
        pos1, neg1, _, pos2, neg2 = problem.split('-')
        X_s, y_s = get_XY(feats[types[0]][pos1], feats[types[1]][neg1])
        X_t, y_t = get_XY(feats[types[0]][pos2], feats[types[1]][neg2])
        src = data_holder(X_s, y_s)
        targ = data_holder(X_t, y_t)
        acc[i] = train_and_test(src, targ, Ws, lap['%s-%s'%(pos2, neg2)])
        last_end = time.time()
        print 'Takes %4.2f s, accuracy is %3.4f'%(last_end-last_start, acc[i])
        last_start = last_end
        with open(option.out_file, 'wb') as f:
            np.savetxt(f, acc*100, delimiter=',', fmt='%.4f')
    duration = time.time() - start
    print 'Average accuracy is ', np.mean(acc)
    print '%d problems take %f s'%(len(problems), duration)

if __name__ == '__main__':
    main()
