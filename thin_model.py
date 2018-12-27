    import tensorflow as tf
import numpy as np
from tqdm import *
import image_utils
import math

def sort_pool2d(x, pool_weights, k=4, padding='SAME'):
  assert k in [1,2,3,4]
  if k==1:
    return tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], padding=padding)
  batch_size, height, width, num_channels = x.get_shape().as_list()
  pad_bottom = height%2
  pad_right = width%2
  height_div2 = height + pad_bottom
  width_div2 = width + pad_right
  if(padding=='SAME'):
    x = tf.pad(x, [[0, 0], [0, pad_bottom], [0, pad_right], [0, 0]], "CONSTANT")

  _, height, width, _ = x.get_shape().as_list()
  offsets_y = tf.range(0, height, 2)
  offsets_x = tf.range(0, width, 2)

  sub_y0 = tf.gather(x, offsets_y, axis=1)
  sub_y1 = tf.gather(x, offsets_y + 1, axis=1)

  sub_00 = tf.gather(sub_y0, offsets_x, axis=2)
  sub_00 = tf.reshape(sub_00, [-1])
  sub_01 = tf.gather(sub_y0, offsets_x + 1, axis=2)
  sub_01 = tf.reshape(sub_01, [-1])

  sub_10 = tf.gather(sub_y1, offsets_x, axis=2)
  sub_10 = tf.reshape(sub_10, [-1])
  sub_11 = tf.gather(sub_y1, offsets_x + 1, axis=2)
  sub_11 = tf.reshape(sub_11, [-1])

  sub0 = tf.where(tf.less(sub_00, sub_01), tf.stack([sub_00, sub_01], axis=1),
                  tf.stack([sub_01, sub_00], axis=1))
  sub1 = tf.where(tf.less(sub_10, sub_11), tf.stack([sub_10, sub_11], axis=1),
                  tf.stack([sub_11, sub_10], axis=1))

  sub00 = tf.squeeze(tf.slice(sub0, [0, 0], [-1, 1]))
  sub01 = tf.squeeze(tf.slice(sub0, [0, 1], [-1, 1]))

  sub10 = tf.squeeze(tf.slice(sub1, [0, 0], [-1, 1]))
  sub11 = tf.squeeze(tf.slice(sub1, [0, 1], [-1, 1]))

  # assume elem1 <= elem3
  def sort_elems(elem1, elem2, elem3, elem4):
    elem2_less_than_elem3 = tf.stack([elem1, elem2, elem3, elem4], axis=1)
    elem2_greater_equal_elem3_and_elem2_less_than_elem4 = tf.stack([elem1, elem3, elem2, elem4], axis=1)
    elem2_greater_equal_elem3_and_elem2_greater_equal_elem4 = tf.stack([elem1, elem3, elem4, elem2], axis=1)
    elem2_greater_equal_elem3 = tf.where(tf.less(elem2, elem4),
      elem2_greater_equal_elem3_and_elem2_less_than_elem4, elem2_greater_equal_elem3_and_elem2_greater_equal_elem4)
    return tf.where(tf.less(elem2, elem3), elem2_less_than_elem3, elem2_greater_equal_elem3)

  sub00_less_sub10 = sort_elems(sub00, sub01, sub10, sub11)
  sub00_greater_equal_sub10 = sort_elems(sub10, sub11, sub00, sub01)

  sorted_sub_flat = tf.where(tf.less(sub00, sub10), sub00_less_sub10, sub00_greater_equal_sub10)
  sorted_sub = tf.slice(sorted_sub_flat, [0, 4-k], [-1, k])
  sorted_sub = tf.reshape(sorted_sub, [-1, int(height/2), int(width/2), num_channels, k])

  #with tf.variable_scope(name):
  #pool_weights = tf.get_variable('pool_weights', [1,1,1,1,k], tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
  pool_weights = tf.nn.softmax(pool_weights)

  weighted_subsets = pool_weights*sorted_sub
  x = tf.reduce_sum(weighted_subsets, 4)
  return x

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    tf.add_to_collection("weights", initial)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.3, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# ksize = 3,3 for overlapping pooling
def max_pool_2x2(x, W):
    #return sort_pool2d(x, W)
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#axes=[0,1,2]
def normalize_batch(X, gamma, beta, axes=[0,1,2]):
    mean, var = tf.nn.moments(X, axes=axes, keep_dims=False)
    return tf.nn.batch_normalization(X, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=1e-5)

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

CONV = 'CONV'
POOL = 'POOL'
FC = 'FC'
DROPOUT = 'DROPOUT'
RELU = 'RELU'
NORM = 'NORM'
RES_BEG = 'RES_BEG'
RES_END = 'RES_END'
RES_BEG1 = 'RES_BEG1'
RES_BEG2 = 'RES_BEG2'
RES_END1 = 'RES_END1'
RES_END2 = 'RES_END2'
classes = 10

def build_layers(layers):
    model_layers = []
    current_shape = [3,24,24]
    for layer in layers:
        layer_dict = {}
        layer_dict['name'] = layer['name']
        if layer['name'] == CONV:
            layer_dict['W'] = weight_variable(shape =[
                                                layer['shape'][0],
                                                layer['shape'][1],
                                                current_shape[0],
                                                layer['shape'][2]],
                                            stddev=math.sqrt(2.0/(9.0*current_shape[0])))
            layer_dict['b'] = bias_variable([layer['shape'][2]])
            current_shape[0] = layer['shape'][2]
        elif layer['name'] == FC:
            layer_dict['W'] = weight_variable([np.prod(current_shape), layer['shape'][0]])
            layer_dict['b'] = bias_variable([layer['shape'][0]])
            layer_dict['W_params'] = np.prod(current_shape)
            current_shape = layer['shape']
        elif layer['name'] == POOL:
            layer_dict['W'] = weight_variable([1,1,1,1,4])
            current_shape[1] = current_shape[1]//2
            current_shape[2] = current_shape[2]//2
        elif layer['name'] == NORM:
            layer_dict['mean'] = tf.Variable(tf.random_normal([current_shape[0]]))
            layer_dict['var'] = tf.Variable(tf.random_normal([current_shape[0]]))+1
            if 'axes' in layer:
                layer_dict['axes'] = layer['axes']
        # in case of dropout change nothing
        layer_dict['id'] = layer['id']
        model_layers.append(layer_dict)
    return model_layers

def crop_wrapper(image):
    return tf.random_crop(image, [24,24,3])

def central_crop_wrapper(image):
    return tf.random_crop(image, [24,24,3])

def contrast_wrapper(image):
    return tf.image.random_contrast(image,lower=0.2, upper=1.8)

def brightness_wrapper(image):
    return tf.image.random_brightness(image, max_delta=63)

def model(X, layers):
    X = tf.cond(is_training,
                    lambda:tf.map_fn(tf.image.per_image_standardization, X),
                    lambda:tf.identity(X))
    X = tf.cond(is_training,
                    lambda:tf.map_fn(tf.image.random_flip_left_right, X),
                    lambda:tf.identity(X))
    X = tf.cond(is_training,
                    lambda:tf.map_fn(contrast_wrapper, X),
                    lambda:tf.identity(X))

    temp = []
    temp1 = []
    temp2 = []
    for layer in layers:
        # drop layer
        layer_id = tf.constant(layer['id'], dtype=tf.int32)
        if layer['name'] == CONV:
            X = conv2d(X, layer['W']) + layer['b']
        elif layer['name'] == POOL:
            X = max_pool_2x2(X, layer['W'])
        elif layer['name'] == FC:
            X = tf.reshape(X, [-1, layer['W_params']])
            X = tf.matmul(X, layer['W']) + layer['b']
        elif layer['name'] == DROPOUT:
            X = tf.nn.dropout(X, keep_probs)
        elif layer['name'] == RELU:
            X = tf.nn.relu(X)
        elif layer['name'] == NORM:
            if 'axes' in layer:
                X = normalize_batch(X, layer['var'], layer['mean'], axes = layer['axes'])
            else:
                X = normalize_batch(X, layer['var'], layer['mean'])
        elif layer['name'] == RES_BEG:
            temp = X
        elif layer['name'] == RES_END:
            X = X + temp
        elif layer['name'] == RES_BEG1:
            temp1 = X
        elif layer['name'] == RES_END1:
            X = X + temp1
        elif layer['name'] == RES_BEG2:
            temp2 = X
        elif layer['name'] == RES_END2:
            X = X + temp2
    return X

layers = [
    {'name':NORM, "id":0},

    {'name':CONV, 'shape':[5,5,16], "id":0},#32

    {'name':RES_BEG1, "id":1},

    {'name':RES_BEG, "id":0},
    {'name':NORM, "id":0},
    {'name':RELU, "id":0},
    {'name':CONV, 'shape':[5,5,16], "id":0},#16
    {'name':NORM, "id":0},
    {'name':RELU, "id":0},
    {'name':CONV, 'shape':[5,5,16], "id":0},#16
    {'name':RES_END, "id":0},

    {'name':RES_BEG, "id":0},
    {'name':NORM, "id":0},
    {'name':RELU, "id":0},
    {'name':CONV, 'shape':[3,3,16], "id":0},#16
    {'name':NORM, "id":0},
    {'name':RELU, "id":0},
    {'name':CONV, 'shape':[3,3,16], "id":0},#16
    {'name':RES_END, "id":0},

    {'name':RES_END1, "id":1},

    {'name':POOL, "id":0},

    {'name':NORM, "id":1},
    {'name':RELU, "id":1},
    {'name':CONV, 'shape':[3,3,32], "id":1},

    {'name':RES_BEG, "id":1},
    {'name':NORM, "id":1},
    {'name':RELU, "id":1},
    {'name':CONV, 'shape':[3,3,32], "id":1},
    {'name':NORM, "id":1},
    {'name':RELU, "id":1},
    {'name':CONV, 'shape':[3,3,32], "id":1},
    {'name':RES_END, "id":1},

    {'name':POOL, "id":0},

    {'name':RES_BEG, "id":6},
    {'name':NORM, "id":6},
    {'name':RELU, "id":6},
    {'name':CONV, 'shape':[3,3,32], "id":6},
    {'name':NORM, "id":6},
    {'name':RELU, "id":6},
    {'name':CONV, 'shape':[3,3,32], "id":6},
    {'name':RES_END, "id":6},

    {'name':POOL, "id":0},

    {'name':RES_BEG, "id":8},
    {'name':NORM, "id":8},
    {'name':RELU, "id":8},
    {'name':CONV, 'shape':[3,3,32], "id":8},
    {'name':NORM, "id":8},
    {'name':RELU, "id":8},
    {'name':CONV, 'shape':[3,3,32], "id":8},
    {'name':RES_END, "id":8},

    {'name':NORM, "id":0},
    {'name':RELU, "id":0},
    {'name':DROPOUT, "id":0},
    {'name':FC, 'shape':[512], "id":0},
    {'name':NORM, 'axes':[0], "id":0},
    {'name':RELU, "id":0},
    {'name':FC, 'shape':[10], "id":0}
]
saver_name = 'saver/saver2.ckpt'

x = tf.placeholder(tf.float32, [None, 24,24,3])
y = tf.placeholder(tf.float32, [None, classes])
keep_probs = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

def _cost_op(model):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y)
    )

# returns part of set (of batch size) and batch size
#(because may not be equal to batch_size in case of last iteration)
def batch_iterator(_set, batch_size):
    total_batch = int(len(_set)/batch_size)
    for i in range(total_batch):
        batch = _set[i*batch_size:(i+1)*batch_size]
        yield (batch, batch_size)
    if total_batch*batch_size < len(_set):
        yield (_set[total_batch*batch_size:], len(_set)-total_batch*batch_size)

class Model:
    def __init__(self, test=False):
        self.layer_model = None
        if test:
            self.layer_model = model(x, build_layers(test_layers))
        else:
            self.layer_model = model(x, build_layers(layers))

        self.cost_op = _cost_op(self.layer_model)
        self.optimizer_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost_op)
        self.ema = tf.train.ExponentialMovingAverage(decay=0.995)
        var_avg = self.ema.apply(tf.trainable_variables())
        self.ema_saver = tf.train.Saver(self.ema.variables_to_restore())
        self.saver = tf.train.Saver()
        self.training_op = tf.group(self.optimizer_op, var_avg)

    # predicted labels op
    def _predict_op(self):
        return tf.argmax(self.layer_model, 1)

    # return array [0,1,1,0,...] where 1 is equal
    def _count_equal_op(predicted):
        return tf.equal(predicted, tf.argmax(y, 1))

    def count_accuracy(self, images, labels, session):
        batch_size=100
        total_batch = len(labels) // batch_size
        images_iter = batch_iterator(images, batch_size=batch_size)
        labels_iter = batch_iterator(labels, batch_size=batch_size)
        eq_sum = 0.
        self.saver.save(session, saver_name)
        self.ema_saver.restore(session, saver_name)
        # +1 for savety
        for i in tqdm(range(total_batch+1)):
            try:
                batch_images, _ = next(images_iter)
                batch_labels, _ = next(labels_iter)
                #crops = [[0,0],[0,8],[8,0],[8,8],[4,4]]
                crops = [[0,0]]
                outputs = np.zeros((batch_size, 10))
                for crop in crops:
                    cropped_images = image_utils.crop(batch_images, crop[0], crop[1])
                    predictions = self.layer_model.eval({
                        x: cropped_images,
                        keep_probs: 1.0,
                        is_training: False
                    })
                    outputs = outputs + predictions
                avg_outputs = outputs / len(crops)
                add_equal = tf.reduce_sum(tf.cast(
                    tf.equal(tf.argmax(avg_outputs, axis=1), tf.argmax(y, 1)), tf.float32)).eval({
                        y: batch_labels,
                })
                eq_sum += add_equal
            except StopIteration:
                self.saver.restore(session, saver_name)
                return eq_sum / len(labels)

    def train(self, images, labels, session):
        batch_size = 32
        total_batch = int(len(images)/batch_size)
        shuffle_in_unison(images, labels)
        images_iter = batch_iterator(images, batch_size=batch_size)
        labels_iter = batch_iterator(labels, batch_size=batch_size)
        avg_cost = 0.
        for i in tqdm(range(total_batch)):
            try:
                batch_images, _ = next(images_iter)
                batch_labels, _ = next(labels_iter)
                batch_images = image_utils.crop_images(batch_images)
                _, c = session.run([self.training_op, self.cost_op], feed_dict={
                    x: batch_images,
                    y: batch_labels,
                    keep_probs: 0.7,
                    is_training: True,
                })
                avg_cost += c / total_batch
            except StopIteration:
                pass
        print("Cost:", avg_cost)

def get_model(test=False):
    return Model(test)
