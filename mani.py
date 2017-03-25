import numpy as np
import numpy.linalg as la
import tensorflow as tf


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


class vgg_upper(object):
    def __init__(self, W_dict, fc_dims=[512, 10]):
        self.fc_dims = fc_dims
        self.Ws = {}
        for i in xrange(1, 5):
            name = 'conv5_' + str(i)
            self.Ws[name] = W_dict[name]

    def vgg_trans(self, h, t, y, Lap):
        for i in xrange(1, 5):
            name = 'conv5_%d'%i
            h, t = self.conv_relu(h, t, name)
        h = self.max_pool(h, 'pool5')
        t = self.max_pool(t, 'pool5')
        h, t = self.fc_relu(h, t, 'fc6', shape=[512, self.fc_dims[0]])
        h, t = self.fc_relu(h, t, 'fc7', shape=[self.fc_dims[0], self.fc_dims[1]])
        y_pred, t = self.fc(h, t, 'fc8', shape=[self.fc_dims[1], 2])
        t = tf.nn.softmax(t)

        # accuracy
        hit = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(hit, tf.float32))

        # loss
        mani = tf.trace(tf.matmul(tf.matmul(t, Lap, transpose_a=True), t))
        xe = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y))
        reg_penalty = tf.nn.l2_loss(tf.trainable_variables()[0])
        for var in tf.trainable_variables()[1:]:
            reg_penalty += tf.nn.l2_loss(var)
        return xe, reg_penalty, mani, accuracy

    def conv_relu(self, h, t, name):
        with tf.name_scope(name):
            kernel = tf.Variable(self.Ws[name]['W'], name="W")
            bias = tf.Variable(self.Ws[name]['b'].reshape((-1, )), name="b")
            h_conv = tf.nn.conv2d(h, kernel, [1, 1, 1, 1], padding='SAME')
            t_conv = tf.nn.conv2d(t, kernel, [1, 1, 1, 1], padding='SAME')
            h_ret= tf.nn.relu(tf.nn.bias_add(h_conv, bias))
            t_ret= tf.nn.relu(tf.nn.bias_add(t_conv, bias))
            return h_ret, t_ret

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def fc(self, h, t, name, shape):
        with tf.name_scope(name):
            initial = tf.truncated_normal(shape, stddev=0.01)
            W = tf.Variable(initial, name="W")
            initial = tf.truncated_normal(shape[1:], stddev=0.01)
            b = tf.Variable(initial, name="b")

            if len(h.get_shape().as_list()) > 2:
                h = tf.reshape(h, shape=[-1, W.get_shape().as_list()[0]])
                t = tf.reshape(t, shape=[-1, W.get_shape().as_list()[0]])
            h_ret = tf.nn.bias_add(tf.matmul(h, W), b)
            t_ret = tf.nn.bias_add(tf.matmul(t, W), b)
            return h_ret, t_ret

    def fc_relu(self, h, t, name, shape):
        h_ret, t_ret = self.fc(h, t, name, shape)
        h_ret = tf.nn.relu(h_ret)
        t_ret = tf.nn.relu(t_ret)
        return h_ret, t_ret
