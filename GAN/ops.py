import tensorflow as tf
import config

def leakyRelu(X):
    return tf.maximum(X, X * 0.2)

def w_var(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.05))

class FC_full:
    def __init__(self, scope_name, in_size, out_size):
        with tf.variable_scope(scope_name):
            self.W = w_var([in_size, out_size])
    def run(self, X, labels, training):
        X = tf.concat((X,labels), 1)
        X = tf.matmul(X, self.W)
        X = tf.layers.batch_normalization(X, training=training)
        return leakyRelu(X)

class FC:
    def __init__(self, scope_name, in_size, out_size):
        with tf.variable_scope(scope_name):
            self.W = w_var([in_size, out_size])
            self.b = w_var([out_size])
    def run(self, X, labels):
        X = tf.concat((X,labels), 1)
        return tf.matmul(X, self.W) + self.b

class Conv:
    def __init__(self, scope_name, kernel_shape, stride, norm=True):
        with tf.variable_scope(scope_name):
            self.stride = stride
            self.kernel = w_var(kernel_shape)
    def run(self, X, labels):
        labels = tf.reshape(labels, [config.batch_size,1,1,10])
        labels = tf.ones((tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2], 10)) * labels
        X = tf.concat((X, labels), axis=3)
        X = tf.nn.conv2d(X, self.kernel, strides=[1, self.stride, self.stride, 1], padding='SAME')
        X = tf.layers.batch_normalization(X, training=True)
        return leakyRelu(X)

class Deconv:
    def __init__(self, scope_name, kernel_shape, output_shape, stride, tanh=False, norm=True):
        with tf.variable_scope(scope_name):
            self.stride = stride
            self.output_shape = output_shape
            self.kernel = w_var(kernel_shape)
            self.tanh = tanh
            self.norm = norm
    def run(self, X, labels, training):
        labels = tf.reshape(labels, [config.batch_size,1,1,10])
        labels = tf.ones((tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2], 10)) * labels
        X = tf.concat((X, labels), axis=3)
        X = tf.nn.conv2d_transpose(X, self.kernel, self.output_shape, strides=[1, self.stride, self.stride, 1], padding='SAME')
        if self.norm:
            X = tf.layers.batch_normalization(X,training=training)
        if self.tanh:
            return X
        else:
            return tf.nn.relu(X)

class Dropout:
    def __init__(self):
        pass
    def run(self, X, labels):
        return tf.nn.dropout(X, config.keep_probs)

class Reshape:
    def __init__(self, shape):
        self.shape = shape
    def run(self, X, labels, training=False):
        X = tf.map_fn(lambda x : tf.reshape(x, self.shape), X)
        return X
