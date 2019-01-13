import sys
import argparse
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import cv2
import argparse
import time
import config, utils
import preproc_utils

def operation_wrapper(image, function, name, test_mode, gap=0):
    start = time.time()
    image = function(image)
    end = time.time()
    print(name," time: ", end-start)
    if test_mode:
        utils.show_image(name, image, gap)
    return image

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-f', required=True, type=str, help='graph image file')
parser.add_argument('-t', type=preproc_utils.str2bool, help="Test mode")
parser.add_argument('-g', type=int, help="Gap between operations")

args = parser.parse_args()
filename = args.f
test_mode = args.t
gap = args.g
if test_mode == None:
    test_mode = False

if gap == None:
    gap = 0

image = utils.open_image(filename)

if test_mode:
    utils.show_image("Original", np.array(image), gap)

image = operation_wrapper(image, preproc_utils.threshold, "Threshold", test_mode, gap)

image[:2,:] = image[-2:,:] = image[:,-2:] = image[:,:2] = 255

cv2.imwrite('pics/before_cleaning.png', image)
image = operation_wrapper(image, preproc_utils.noise_cleaning, "Noise cleaning", test_mode, gap)
cv2.imwrite('pics/after_cleaning.png', image)

image = operation_wrapper(image, preproc_utils.closing_wrapper, "Closing", test_mode, gap)

image[:2,:] = image[-2:,:] = image[:,-2:] = image[:,:2] = 255

end = time.time()

print("Total time: ", end-start)
if test_mode:
    utils.show_image("Finish", image)

cv2.imwrite('pics/result.png', image)
cv2.destroyWindow("show")
