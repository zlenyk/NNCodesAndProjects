from tqdm import *
from utils import iterate_dataset, sigmoid, read_data, input_to_one_hot, unique_letters
import pickle
import tensorflow as tf
import numpy as np

letters_int_map = {}
int_letters_map = {}

def get_scope_variable(var, shape):
    with tf.variable_scope("weights") as scope:
        try:
            v = tf.get_variable(var, initializer=tf.truncated_normal(shape, stddev=0.1), )
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var, initializer=tf.truncated_normal(shape, stddev=0.1), )
    return v

def init_maps(data):
    current_ind = len(letters_int_map)
    for c in data:
        if c not in letters_int_map:
            letters_int_map[c] = current_ind
            int_letters_map[current_ind] = c
            current_ind += 1

def text_to_int(input):
    return [letters_int_map[c] for c in input]

def step(W, symbol, h, C, hidden_size, batch_size):
    hs = hidden_size
    lstm_input = tf.concat(axis=1, values=[tf.ones([batch_size,1]),symbol,h])
    IFOG = tf.matmul(lstm_input, W)
    IFOGf = tf.concat((tf.sigmoid(IFOG[:,:3*hs]),tf.tanh(IFOG[:,3*hs:])),axis=1)
    C = IFOGf[:,:hs] * IFOGf[:,3*hs:] + IFOGf[:,hs:2*hs] * C
    output = IFOGf[:,2*hs:3*hs] * tf.tanh(C)
    return output, C

def get_probs(predict, h):
    # dense to alphabet
    predictions = tf.matmul(h, predict)
    return tf.subtract(predictions, tf.reduce_logsumexp(predictions, axis=1, keep_dims=True))

def cost_op(logprobs, alphabet_size, batch_size, num_steps):
    one_hot_targets = tf.one_hot(tf.transpose(Y), alphabet_size)
    probs = tf.multiply(logprobs, one_hot_targets)
    divisor_cast = tf.cast(tf.multiply(batch_size, num_steps), tf.float32)
    probs = tf.div(tf.reduce_sum(probs), divisor_cast)
    return tf.negative(probs)

X = tf.placeholder(tf.int32)
Y = tf.placeholder(tf.int32)
H = tf.placeholder(tf.float32)
C = tf.placeholder(tf.float32)

class RNN:
    def __init__(self, hidden_size, alphabet_size, batch_size=1, num_steps=1):

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_steps = num_steps

        self.W = get_scope_variable("W",[hidden_size + hidden_size + 1, 4 * hidden_size])
        self.linear = get_scope_variable("lin", [alphabet_size, hidden_size])

        one_hot_input = tf.one_hot(X, alphabet_size)
        logprobs = []
        h = H
        c = C
        for i in range(num_steps):
            symbol_in = one_hot_input[:, i]
            symbol_in = tf.matmul(symbol_in, self.linear)
            h, c = step(self.W, symbol_in, h, c, hidden_size, batch_size)
            probs = get_probs(tf.transpose(self.linear), h)
            logprobs.append(probs)
        self.logprobs = tf.stack(logprobs)
        self.loss_op = cost_op(self.logprobs, alphabet_size, batch_size, num_steps)
        self.training_op = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(self.loss_op)
        self.h = h
        self.c = c

def generate_text(text):
    model = RNN(batch_size=1, num_steps=1, hidden_size=200, alphabet_size=alphabet_size)
    h = np.zeros((model.batch_size, model.hidden_size))
    c = np.zeros((model.batch_size, model.hidden_size))
    for symbol in text_to_int(text):
        probs, h, c = sess.run([model.logprobs, model.h, model.c], feed_dict={
            X: np.atleast_2d(symbol),
            H: h,
            C: c
        })
    symbols = []
    for i in range(1000):
        # chooses 1 val from 0 to 82, with probability p
        symbol = np.random.choice(len(letters_int_map), p=np.exp(probs[0][0]))
        text += int_letters_map[symbol]
        probs, h, c = sess.run([model.logprobs, model.h, model.c], feed_dict={
            X: np.atleast_2d(symbol),
            H: h,
            C: c
        })
    print(text)

def validate(data, alphabet_size):
    model = RNN(batch_size=1, num_steps=1, hidden_size=200, alphabet_size=alphabet_size)
    data = text_to_int(data)
    data_iterator = iterate_dataset(data, model.batch_size, model.num_steps)
    batch_length = len(valid_data) // model.batch_size
    h = np.zeros((model.batch_size, model.hidden_size))
    c = np.zeros((model.batch_size, model.hidden_size))
    losses = []
    iters = 0
    for j in tqdm(range(batch_length)):
        try:
            iters += 1
            (x, y) = next(data_iterator)
            loss, h, c = sess.run([model.loss_op, model.h, model.c], feed_dict={
                X: x,
                Y: y,
                H: h,
                C: c
            })
            losses.append(loss)
        except StopIteration:
            print("Validation perplexity:", np.exp(np.mean(losses)))
            break

def train(sess, data, alphabet_size):
    model = RNN(batch_size=20, num_steps=15, hidden_size=200, alphabet_size=alphabet_size)
    data = text_to_int(data)
    batch_length = len(data) // model.batch_size
    epoch_size = (batch_length - 1) // model.num_steps
    data_iterator = iterate_dataset(data, model.batch_size, model.num_steps)
    h = np.zeros((model.batch_size, model.hidden_size))
    c = np.zeros((model.batch_size, model.hidden_size))
    for i in tqdm(range(epoch_size)):
        try:
            (x, y) = next(data_iterator)
            _, h, c, loss = sess.run([model.training_op, model.h, model.c, model.loss_op], feed_dict={
                X: x,
                Y: y,
                H: h,
                C: c
            })
        except StopIteration:
            return

num_epochs = 10
train_data = (read_data('pan_tadeusz_1_10.txt'))
valid_data = (read_data('pan_tadeusz_11.txt'))
alphabet_size = unique_letters(train_data+valid_data)
init_maps(train_data+valid_data)
RNN(hidden_size=200, alphabet_size=alphabet_size)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

for j in range(num_epochs):
    train(sess, train_data, alphabet_size)
    print("After Epoch " + str(j))
    validate(valid_data, alphabet_size)
    generate_text("Jam jest Jacek")
