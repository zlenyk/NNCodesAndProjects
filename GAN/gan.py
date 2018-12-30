from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tqdm import *
import numpy as np
import utils
import cifar_input
import config
from ops import FC_full, FC, Deconv, Conv, Reshape, Dropout
import math
import time
import pickle


class Generator:
    def __init__(self):
        self.layers = [
            FC_full("g", 100+10, 4*4*256),
            Reshape([4,4,256]),
            Deconv("g", kernel_shape=[3,3,128,266], output_shape=[config.batch_size,8,8,128], stride=2),
            Deconv("g", [3,3,128,138], [config.batch_size,16,16,128], 2),
            Deconv("g", [3,3,3,138], [config.batch_size,32,32,3], 2, tanh=True, norm=False),
        ]
    def run(self, X, labels, training=False):
        for i in range(len(self.layers)):
            X = self.layers[i].run(X, labels, training=training)
        return tf.nn.tanh(X)

# label = [1,1,1,10]
# y = tf.ones( tf.shape(x)[0], ..., )* label
# tf.concat([x,y], 3)
class Discriminator:
    def __init__(self):
        self.layers = [
            Conv("d", [3,3,13,128], 1),#32
            Conv("d", [3,3,138,128], 2),#16
            Dropout(),
            Conv("d", [3,3,138,128], 1),
            Conv("d", [3,3,138,128], 2),#8
            Dropout(),
            Conv("d", [3,3,138,128], 1),
            Conv("d", [3,3,138,128], 2),#4
            Reshape([128*4*4]),
            FC("d", 128*4*4+10, 1),
        ]
    def run(self, X, labels):
        for i in range(len(self.layers)):
            X = self.layers[i].run(X, labels)
        return X

x = tf.placeholder(tf.float32, [None, 32,32,3])
real_labels = tf.placeholder(tf.float32, [None, 10])
random_seed = tf.placeholder(tf.float32, [None, 100])
fake_labels = tf.placeholder(tf.float32, [None, 10])
training = tf.placeholder(tf.bool)

def samples(gan, name):
    batch, one_hot_labels = utils.get_samples_seeds()
    gener = gan.generations.eval({
        random_seed: batch,
        fake_labels: one_hot_labels,
        training: False,
    })
    gener = (gener+1)/2
    utils.mult_plot(gener, one_hot_labels, name)

def draw_random_images(gan, epoch):
    training_size = 50000
    print("Generating images")
    images = []
    labels = []
    for j in range(int(training_size/config.batch_size)):
        batch, one_hot_labels = utils.get_random_seeds()
        gener = gan.generations.eval({
            random_seed: batch,
            fake_labels: one_hot_labels,
            training: False,
        })
        gener = (gener+1)/2
        gener *= 255
        gener = np.uint8(gener)
        images = images + gener.tolist()
        batch_labels = np.argmax(one_hot_labels, axis=1)
        labels = labels + batch_labels.tolist()

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels)
    np.save('gan_images.npy', images)
    np.save('gan_labels.npy', labels)
    print("Generated")

class GAN:
    #ph: random_seed [batch_size, 100]
    #   fake_labels [batch_size, 10]g
    #   x [batch_size, 784]
    #   real_labels [batch_size, 10]
    def __init__(self):
        discriminator = Discriminator()
        generator = Generator()

        self.normX = 2*x - 1
        self.generations = generator.run(random_seed, fake_labels, training=training)
        D_real = discriminator.run(self.normX, real_labels)
        D_fake = discriminator.run(self.generations, fake_labels)

        random = tf.random_uniform(tf.shape(real_labels), minval=1, maxval=10, dtype=tf.float32, seed=None, name=None)
        bad_labels = real_labels + random
        bad_labels = tf.floormod(bad_labels, tf.fill(tf.shape(D_fake), 10.0))
        D_wrong = discriminator.run(self.normX, bad_labels)

        self.gen_cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.fill(tf.shape(D_fake), 0.9), name="GEN_COST"))

        self.dis_cost_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.fill(tf.shape(D_fake), 0.9), name="DIS_COST_REAL"))
        self.dis_cost_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros(tf.shape(D_fake)), name="DIS_COST_FAKE"))
        self.dis_cost_bad_labels = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_wrong, labels=tf.zeros(tf.shape(D_fake)), name="DIS_COST_WRONG"))

        self.dis_cost = self.dis_cost_real + self.dis_cost_fake + self.dis_cost_bad_labels

        #self.dis_real_opt = tf.train.AdamOptimizer(learning_rate=0.0003, beta1=0.5).minimize(self.dis_cost_real,
        #    var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='d'))
        #self.dis_fake_opt = tf.train.AdamOptimizer(learning_rate=0.0003, beta1=0.5).minimize(self.dis_cost_fake,
        #    var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='d'))

        self.dis_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.dis_cost,
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='d'))
        self.gen_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.gen_cost,
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='g'))

def train_gan():
    gan = GAN()
    sess = tf.InteractiveSession()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('summaries', sess.graph)

    tf.global_variables_initializer().run()
    (train_images, train_labels), _ = cifar_input.import_cifar()

    #pre-train discriminator
    images_iter = utils.batch_iterator(train_images, batch_size=config.batch_size)
    labels_iter = utils.batch_iterator(train_labels, batch_size=config.batch_size)
    for i in range(0):
        batch_x, _ = next(images_iter)
        batch_y, _ = next(labels_iter)
        batch, one_hot_labels = utils.get_random_seeds()
        _, d_c_real, d_c_fake = sess.run([gan.dis_opt, gan.dis_cost_real, gan.dis_cost_fake], feed_dict={
            x: batch_x,
            real_labels: batch_y,
            random_seed: batch,
            fake_labels: one_hot_labels,
            training: True,
        })

    for epoch in range(50):

        total_batch = int(len(train_images)/config.batch_size)

        start = time.time()
        images_iter = utils.batch_iterator(train_images, batch_size=config.batch_size)
        labels_iter = utils.batch_iterator(train_labels, batch_size=config.batch_size)

        stats_array = []

        d_c_real = d_c_fake = g_c = 2.0
        print("Epoch " + str(epoch))
        for i in tqdm(range(total_batch)):
            batch_x, _ = next(images_iter)
            batch_y, _ = next(labels_iter)
            batch, one_hot_labels = utils.get_random_seeds()

            _, d_c_real, d_c_fake, d_bad = sess.run([gan.dis_opt, gan.dis_cost_real, gan.dis_cost_fake, gan.dis_cost_bad_labels], feed_dict={
                x: batch_x,
                real_labels: batch_y,
                random_seed: batch,
                fake_labels: one_hot_labels,
                training: True,
            })

            disc_real, disc_fake, d_bad = d_c_real, d_c_fake, d_bad

            gen_iter = 0
            for k in range(3):
                _, g_c = sess.run([gan.gen_opt, gan.gen_cost], feed_dict={
                    random_seed: batch,
                    fake_labels: one_hot_labels,
                    training: True,
                })
            gen_end = g_c

            stats_array.append({'d_r':disc_real, 'd_f': disc_fake, 'd_bad': d_bad, 'g_e': gen_end})

        disc_real = [d['d_r'] for d in stats_array]
        disc_fake = [d['d_f'] for d in stats_array]
        d_bad_labels = [d['d_bad'] for d in stats_array]
        gen_end = [d['g_e'] for d in stats_array]

        output_string_max = 'MAX Disc real {:.2f} Disc fake {:.2f} Disc bad labels {:.2f} Generator {:.2f}'.format(
                    np.amax(disc_real),
                    np.amax(disc_fake),
                    np.amax(d_bad_labels),
                    np.amax(gen_end))
        output_string_avg = 'AVG Disc real {:.2f} Disc fake {:.2f} Disc bad labels {:.2f} Generator {:.2f}'.format(
                    np.average(disc_real),
                    np.average(disc_fake),
                    np.average(d_bad_labels),
                    np.average(gen_end))
        output_string_min = 'MIN Disc real {:.2f} Disc fake {:.2f} Disc bad labels {:.2f} Generator {:.2f}'.format(
                    np.amin(disc_real),
                    np.amin(disc_fake),
                    np.amin(d_bad_labels),
                    np.amin(gen_end))

        print(output_string_max)
        print(output_string_avg)
        print(output_string_min)
        stats_array = []

        samples(gan, str(epoch)+"_"+str(i))
        if epoch%5 == 4:
            draw_random_images(gan, epoch)

train_gan()
