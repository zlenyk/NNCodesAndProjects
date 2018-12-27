#All the code written myself (ZYgmunt Lenyk)
import cifar_input
import cifar_model
import tensorflow as tf
from skimage import transform
import sys
import math
import numpy as np
import utils
import argparse

display_step = 1
batch_size = 32
num_epochs = 50
classes = 10

parser = argparse.ArgumentParser()
parser.add_argument('-g', type=utils.str2bool, help="GAN mode")
args = parser.parse_args()
gan_mode = args.g
if gan_mode == None:
    gan_mode = False

if gan_mode:
    (train_images, train_labels), test_set = cifar_input.import_cifar_gan()
else:
    (train_images, train_labels), test_set = cifar_input.import_cifar()

print(train_images.shape, train_labels.shape)
models_probs = np.empty((0,len(test_set[0]),classes))
models_labels = []

sess = tf.InteractiveSession()
saver_name = 'saver/saver2.ckpt'

validation_size = 10000
for k in range(5):
    model = cifar_model.get_model(test=False)
    tf.global_variables_initializer().run()
    model.saver.save(sess, saver_name)
    for epoch in range(num_epochs):
        model.train(train_images, train_labels, sess)

        if epoch % display_step == 0:
            (test_images, test_labels) = test_set
            valid_images = train_images[:validation_size]
            valid_labels = train_labels[:validation_size]
            print("Model:", k, "Epoch:", '%04d' % (epoch+1),
                #"Validation accuracy:", model.count_accuracy(valid_images, valid_labels, sess),
                "Test accuracy:", model.count_accuracy(test_images, test_labels, sess))
