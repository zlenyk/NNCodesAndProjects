import autograd.numpy as np

def sigmoid(x):
    return 0.5*(np.tanh(x) + 1.0)   # Output ranges from 0 to 1.

def iterate_dataset(raw_data, batch_size, num_steps):
    batch_length = len(raw_data) // batch_size
    epoch_size = (batch_length - 1) // num_steps
    data = np.zeros([batch_size, batch_length], dtype=np.int32)

    for i in range(batch_size):
        data[i] = raw_data[i*batch_length : (i+1)*batch_length]

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i+1) * num_steps]
        y = data[:, i * num_steps+1:(i+1) * num_steps + 1]
        yield (x, y)

def read_data(filename):
    with open(filename, encoding='utf-8-sig', mode='U') as f:
        return f.read()

# changes array shape to array shape + [alphabet size] - one hot for each element
def input_to_one_hot(input, alphabet_size):
    one_hot_shape = input.shape + (alphabet_size,)
    one_hot = np.zeros(one_hot_shape)
    for idx, val in np.ndenumerate(input):
        one_hot[idx][val] = 1
    return one_hot

def unique_letters(input):
    return len(set(input))
