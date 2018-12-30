import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import cifar_input
import numpy as np
import config

def get_random_seeds(batch_size = config.batch_size, lab=None):
    lab = np.random.randint(10, size=batch_size)
    one_hot_labels = np.zeros([batch_size, 10])
    one_hot_labels[np.arange(batch_size), lab] = 1
    return np.random.normal(0.0, 0.1, (batch_size, 100)), one_hot_labels

def get_samples_seeds():
    lab = np.zeros([config.batch_size])
    for i in range(10):
        lab[10*i:10*(i+1)] = i
    lab = lab.astype(int)
    one_hot_labels = np.zeros([100, 10])
    one_hot_labels[np.arange(100), lab] = 1
    return np.random.normal(0.0, 0.1, (100, 100)), one_hot_labels

def mult_plot(images, labels, name):
    rows = 6
    classes = 10
    fig, ax = plt.subplots(rows,classes)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0.1)
    names = cifar_input.get_labels()
    for i in range(rows):
        for j in range(classes):
            label = names[np.argmax(labels[classes*j+i])]
            ax[i][j].axis('off')
            ax[i][j].imshow(images[classes*j+i].reshape(32,32,3), interpolation='nearest', vmin=0.0, vmax=1.0)
            if i == 0:
                ax[i][j].set_title(label, fontsize=8)
    plt.savefig('../results/result_'+name+'.png')
    plt.close(fig)

def get_class_indices(_set):
    names = cifar_input.get_labels()
    labels_classes = np.argwhere(_set[1] == 1)
    classes_indices = []
    for i in range(classes):
        class_indices = np.argwhere(labels_classes[:,1] == i).ravel()
        classes_indices.append(class_indices)
    return np.asarray(classes_indices)

def batch_iterator(_set, batch_size):
    total_batch = int(len(_set)/batch_size)
    for i in range(total_batch):
        batch = _set[i*batch_size:(i+1)*batch_size]
        yield (batch, batch_size)
    if total_batch*batch_size < len(_set):
        yield (_set[total_batch*batch_size:], len(_set)-total_batch*batch_size)
