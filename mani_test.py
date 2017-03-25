import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tensorflow as tf
from cifar_data import *
from mani import *
from vgg import load_vgg_weights, extract_vgg_feature


data = load_data()
vgg_W, mean_im = load_vgg_weights()
types = ["fruit_and_vegetables", "household_electrical_devices"]

feat = {}
for super_c in data:
    feat[super_c] = {}
    for c in data[super_c]:
        feat[super_c][c] = extract_vgg_feature(data[super_c][c] - mean_im, vgg_W)
        
problems = []
i = 0
for pos1 in data[types[0]]:
    for pos2 in data[types[0]]:
        if pos1 == pos2:
            continue
        for neg1 in data[types[1]]:
            for neg2 in data[types[1]]:
                if neg1 == neg2:
                    continue
                name = "%s-%s-vs-%s-%s"%(pos1, neg1, pos2, neg2)
                problems.append(name)

# train_problems = np.random.choice(problems, 4)
train_problems = ['mushroom-lamp-vs-apple-keyboard', 'pear-lamp-vs-mushroom-keyboard',
                'orange-television-vs-mushroom-lamp','sweet_pepper-clock-vs-pear-telephone']

def train(src, targ, Ws, config={}, pre_dir=None, save_dir='pre/best.ckpt'):
    '''
    src - source domain data
    targ - target domain data
    '''
    # get keywords
    alpha = np.float32(config.pop('alpha', 0.1))
    beta = np.float32(config.pop('beta', 1))
    gamma = np.float32(config.pop('gamma', 0.01))
    lr = config.pop('lr', 1e-5)
    prob = config.pop('keep_prob', 1.)
    max_steps = config.pop('steps', 1000)
    batch_size = config.pop('batch_size', 30)
    epochs_per_decay = config.pop('epochs_per_decay', 10)
    verbose = config.pop('verbose', 0)
    dim = config.pop('dim', 10)
    k = config.pop('k', 10)
    M = get_kNN(targ.images, k)
    Lap = np.diag(np.sum(M, axis=1)) - M
    
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default(), tf.device('/gpu:0'):
        x_targ = tf.placeholder(tf.float32, [None, 2, 2, 512])
        x_src = tf.placeholder(tf.float32, [None, 2, 2, 512])
        y_src = tf.placeholder(tf.float32, [None, 2])
        
        # build the model
        net = vgg_upper(Ws, fc_dims=[512, dim])
        xe, reg_penalty, mani, acc_op = net.vgg_trans(x_src, x_targ, y_src, Lap)
    
        # training op
        loss_op = tf.mul(alpha, mani) + tf.mul(beta, xe) + tf.mul(gamma, reg_penalty)
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(lr, global_step*batch_size, 
                                               epochs_per_decay*src.size, 0.5,staircase=True)
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
    best_loss, best_acc = 10000., 0.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0
    # config.log_device_placement=True
    config.allow_soft_placement=True
    with tf.Session(graph=g, config=config) as sess:
        saver = tf.train.Saver()
        # random init or use pretrained model
        if pre_dir is None:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, pre_dir)
        for i in xrange(max_steps):
            sess.run(train_step, feed_dict={x_src:src.images, y_src:src.labels, x_targ:targ.images})
            if (i+1)%verbose == 0 or i+1 == max_steps or i == 0:
                train_loss = sess.run(xe, feed_dict={x_src:src.images, y_src:src.labels})
                train_acc = sess.run(acc_op, feed_dict={x_src:src.images, y_src:src.labels})
                targ_loss = sess.run(xe, feed_dict={x_src:targ.images, y_src:targ.labels})
                targ_acc = sess.run(acc_op, feed_dict={x_src:targ.images, y_src:targ.labels})
                mnf = sess.run(mani, feed_dict={x_targ:targ.images})
                if verbose > 0 and ((i+1)%verbose == 0 or i+1 == max_steps or i == 0):
                    print '%4d/%4d loss/acc:[src %6f, %6f] [targ %6f %.6f], mani %f'%(i+1, 
                        max_steps, train_loss, train_acc, targ_loss, targ_acc, mnf)      
        saver.save(sess, save_dir)
        print 'Model saved in ', save_dir
    return targ_acc

alphas = np.logspace(-4, -2, 1)
gammas = np.logspace(-2, 0, 1)
# dims = [8, 16, 32, 64, 128, 256, 512]
dims = [512]
knns = [5]
lrs = np.logspace(-2.5, -2.5, 1)
lr = lrs[0]
rr = np.zeros((len(problems), 1), dtype=np.float32)
for k, name in enumerate(problems):
    pos1, neg1, _, pos2, neg2 = name.split('-')
    X_s, y_s = get_XY(feat[types[0]][pos1], feat[types[1]][neg1])
    X_t, y_t = get_XY(feat[types[0]][pos2], feat[types[1]][neg2])
    src = data_holder(X_s, y_s)
    targ = data_holder(X_t, y_t)
    print 'Train classifier for ', name
    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            for d, dim in enumerate(dims):
                for l, knn in enumerate(knns):
                    print '%d / 400, alpha %8f, gamma %8f, knn %d'%(k+1, alpha, gamma, knn)
                    temp = train(src, targ, vgg_W, config={
                            'lr':lr, 'alpha':alpha, 'beta':1, 'gamma':gamma, 'keep_prob':1, 'dim':dim,'steps':200,
                            'verbose':40, 'batch_size':src.size, 'epochs_per_decay':20, 'k':knn
                            }, pre_dir=None, save_dir='ckpt/model_saved.ckpt')
                    print 'Test accuracy is ', temp
                    rr[k, 0] = temp
                    with open('test.csv', 'wb') as f:
                       np.savetxt(f, rr*100, delimiter=',', fmt='%.4f')
print rr
print np.mean(rr, axis=0)
