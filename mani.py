import numpy as np
import os
import numpy.linalg as la
import tensorflow as tf

def load_lap(data, k, types, lap_path='datasets/lap.npy'):
    lap = {}
    if os.path.isfile(lap_path) == False:
        for i, pos in data[types[0]].iteritems():
            for j, neg in data[types[1]].iteritems():
                M = get_kNN(np.concatenate((pos, neg), axis=0), k)
                lap['%s-%s'%(i, j)] = np.diag(np.sum(M, axis=1)) - M
        with open(lap_path, 'wb') as f:
            np.save(f, lap)
    else:
        with open(lap_path, 'rb') as f:
            lap = np.load(f).item()
    return lap

def get_kNN(X, k):
    '''
    This function return the k nearest neighbors of each elements,
    which measured by cosine distance
    '''
    assert len(X.shape) > 1, "X should be at least rank 2"
    if len(X.shape) > 2:
        X = X.reshape((X.shape[0], -1))

    # normalize X
    X_norm = X / la.norm(X, axis=1, keepdims=True)

    # compute cosine distance
    dist = X_norm.dot(X_norm.T)

    # patially sort the dist to find knn
    ind = np.argpartition(dist, -k-1, axis=1)
    M = np.zeros((X.shape[0], X.shape[0]), dtype=np.float32)
    M[np.arange(X.shape[0]).reshape((-1, 1)), ind[:, -k-1:]] = 1.
    M[ind[:, -k-1:], np.arange(X.shape[0]).reshape((-1, 1))] = 1.
    M[xrange(X.shape[0]), xrange(X.shape[0])] = 0.
    return M


class model(object):
    def __init__(self, W_dict):
        self.Ws =W_dict
        self.layers = {}

    def build_higher(self, Xs, Xt, y, Lap):
        # concat source and target domain for handling
        h, t = Xs, Xt
        for i in xrange(1, 5):
            name = 'conv5_%d'%i
            h = self.conv_relu(h, name)
            t = self.conv_relu(t, name)
        h = self.max_pool(h, 'pool5')
        t = self.max_pool(t, 'pool5')
        h = tf.nn.relu(self.fc(h, 'fc6', shape=[512, 512]))
        t = tf.nn.relu(self.fc(t, 'fc6', shape=[512, 512]))
        h = tf.nn.relu(self.fc(h, 'fc7', shape=[512, 512]))
        t = tf.nn.relu(self.fc(t, 'fc7', shape=[512, 512]))
        y_src = self.fc(h, 'fc8', shape=[512, 2])
        y_targ = tf.nn.softmax(self.fc(t, 'fc8', shape=[512, 2])) # normalize it

        # accuracy
        hit = tf.equal(tf.argmax(y_src,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(hit, tf.float32))

        # manifold regularization, cross entropy, L2 regularization term respectively
        mani = tf.trace(tf.matmul(tf.matmul(y_targ, Lap, transpose_a=True), y_targ))
        xe = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_src, labels=y))
        reg_penalty = tf.nn.l2_loss(tf.trainable_variables()[0])
        for var in tf.trainable_variables()[1:]:
            reg_penalty += tf.nn.l2_loss(var)
        return xe, reg_penalty, mani, accuracy

    def build_lower(self, X):
        # conv 1
        h = self.conv_relu(X, 'conv1_1')
        h = self.conv_relu(h, 'conv1_2')
        h = self.max_pool(h, 'pool1')

        # conv2
        h = self.conv_relu(h, 'conv2_1')
        h = self.conv_relu(h, 'conv2_2')
        h = self.max_pool(h, 'pool2')

        #conv 3
        h = self.conv_relu(h, 'conv3_1')
        h = self.conv_relu(h, 'conv3_2')
        h = self.conv_relu(h, 'conv3_3')
        h = self.conv_relu(h, 'conv3_4')
        h = self.max_pool(h, 'pool3')

        #conv4
        h = self.conv_relu(h, 'conv4_1')
        h = self.conv_relu(h, 'conv4_2')
        h = self.conv_relu(h, 'conv4_3')
        h = self.conv_relu(h, 'conv4_4')
        h = self.max_pool(h, 'pool4')

        return h

    def get_variables(self, name, shape=None):
        if name not in self.layers:
            self.layers[name] = {}
            if name[:2] == 'fc':
                assert shape is not None, 'Shape should not be None'
                initial = tf.truncated_normal(shape, stddev=0.01)
                self.layers[name]['W'] = tf.Variable(initial, name="W")
                initial = tf.truncated_normal(shape[1:], stddev=0.01)
                self.layers[name]['b'] = tf.Variable(initial, name="b")
            else:
                self.layers[name]['W'] = tf.Variable(self.Ws[name]['W'], name="W")
                self.layers[name]['b'] = tf.Variable(self.Ws[name]['b'].reshape((-1, )), name="b")
        return self.layers[name]['W'], self.layers[name]['b']


    def conv_relu(self, h, name):
        with tf.name_scope(name):
            kernel, bias = self.get_variables(name)
            h_conv = tf.nn.conv2d(h, kernel, [1, 1, 1, 1], padding='SAME')
            return tf.nn.relu(tf.nn.bias_add(h_conv, bias))

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def fc(self, h, name, shape):
        with tf.name_scope(name):
            W, b = self.get_variables(name, shape=shape)

            if len(h.get_shape().as_list()) > 2:
                h = tf.reshape(h, shape=[-1, W.get_shape().as_list()[0]])
            return tf.nn.bias_add(tf.matmul(h, W), b)
