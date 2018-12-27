import cifar_input
import numpy as np

classes = 10

def get_class_indices(_set):
    names = cifar_input.get_labels()
    labels_classes = np.argwhere(_set[1] == 1)
    classes_indices = []
    for i in range(classes):
        class_indices = np.argwhere(labels_classes[:,1] == i).ravel()
        classes_indices.append(class_indices)
    return np.asarray(classes_indices)

# for parsing bool arguments
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

"""
for j in range(classes):
    print(names[j], " "*(10-len(names[j])), "-",\
        count_accuracy(
            predicted_labels[classes_indices[j]],
            test_labels[classes_indices[j]]))
#saver.restore(tf.get_default_session(), save_name)
"""
"""
#ema_saver.restore(tf.get_default_session(), ema_name)
softmax = tf.nn.softmax(logits=layer_model)
model_probs = np.empty((0,10))
for i in range(len(test_images)//test_batch_size):
probs = softmax.eval({
x: test_images[i*test_batch_size:(i+1)*test_batch_size],
keep_probs: 1.0
})
model_probs = np.concatenate((model_probs,probs))
model_probs = np.asarray(model_probs)
models_probs = np.append(models_probs, np.expand_dims(model_probs,axis=0), axis=0)
worst_probs = np.amin(models_probs, axis=0)
predicted_labels = tf.argmax(worst_probs, axis=1).eval()
print("Joined accuracy:", count_accuracy(predicted_labels, test_labels))
for j in range(classes):
print(names[j], " "*(10-len(names[j])), "-",\
count_accuracy(
predicted_labels[classes_indices[j]],
test_labels[classes_indices[j]]))

print("Optimization Finished!")
log_file.close()
"""
