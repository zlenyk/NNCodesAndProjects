import numpy as np
from scipy import stats, signal

a = np.ones((3,3))
b = np.array([[1,2,3,4],[2,3,4,5],[4,5,6,7],[2,2,2,2]])
print a
print b
b = (np.copy(b)).astype(float)
conv = signal.convolve(b, a, mode='valid')
print conv
