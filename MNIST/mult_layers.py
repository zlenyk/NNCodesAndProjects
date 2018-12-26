#code written myself, exceptions:
# scale_and_rotate_image function (https://piotrmicek.staff.tcs.uj.edu.pl/machine-learning/)
# tensorflow tutorial

#gets to around 98.5% on mnist
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tqdm import *
from utils import transform_batch
import numpy as np

mnist = input_data.read_data_sets("../../MNIST_data/", one_hot=True)
input_size=784
classes=10
x_ph = tf.placeholder(tf.float32, [None, input_size])
y_ph = tf.placeholder(tf.float32, [None, classes])
keep_probs = tf.placeholder(tf.float32)

class MultLayers:
    def __init__(self, layers,learning_rate=0.001):
        all_layers = [input_size] + layers + [classes]
        W = [tf.Variable(tf.random_normal([a,b])) for a,b in zip(all_layers[:-1:], all_layers[1::])]
        b = [tf.Variable(tf.random_normal([a])) for a in all_layers[1::]]
        scales = np.full(len(all_layers)-1, tf.Variable(tf.random_normal([1])) + 1)
        biases = np.full(len(all_layers)-1, tf.Variable(tf.random_normal([1])))

        X = x_ph
        for i in range(len(W)-1):
            X = tf.matmul(X, W[i]) + b[i]
            mean, var = tf.nn.moments(X,[0], keep_dims=True)
            X = tf.nn.batch_normalization(X, mean, var, biases[i], scales[i],1e-5)
            X = tf.nn.relu(X)
            X = tf.nn.dropout(X, keep_probs)
        self.inference_op = tf.matmul(X, W[-1]) + b[-1]
        self.cost_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.inference_op, labels=y_ph))
        self.optimize_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost_op)

display_step = 1
batch_size = 32
num_epochs = 10

sess = tf.InteractiveSession()
model = MultLayers(layers=[500])
tf.global_variables_initializer().run()

for epoch in range(num_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in tqdm(range(total_batch)):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = transform_batch(batch_x, [28,28])
        _, c, outputs = sess.run([model.optimize_op, model.cost_op, model.inference_op], feed_dict={
                                                        x_ph: batch_x,
                                                        y_ph: batch_y,
                                                        keep_probs: 0.8})
    if epoch % display_step == 0:
        predict = tf.equal(tf.argmax(model.inference_op, 1), tf.argmax(y_ph, 1))
        accuracy = tf.reduce_mean(tf.cast(predict, "float"))
        print("Epoch:", '%04d' % (epoch+1),\
                "Accuracy:", \
            accuracy.eval({x_ph: mnist.test.images, y_ph: mnist.test.labels, keep_probs: 1.0}))

print("Optimization Finished!")
