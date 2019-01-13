#import cv2
#import numpy as np
#from PIL import Image, ImageEnhance, ImageFilter, ImageOps
#from thinner import Thinner
#import line_iterator, utils
#from sets import Set
#from scipy import stats, signal, ndimage
#import itertools
#import config
import time
#import graph_draw
#from graphviz import Digraph

start = time.time()

x = 100
y = 100
r = 10
points1 = 0
points2 = 0
for k in range(0):
    for i in range(800):
        for j in range(800):
            if (i-x)*(i-x) + (j-y)*(j-y) < r*r:
                points1 += 1
print("points1: ", points1)

for xx in range(x-r, x+r+1):
    for yy in range(y-r, y+r+1):
        if (xx-x)*(xx-x) + (yy-y)*(yy-y) < r*r:
            points2 += 1
print("points2: ", points2)

end = time.time()
print("Time: " , end-start)
"""
pil_image = Image.open('edges1.jpg').resize((config.img_height, config.img_width), Image.ANTIALIAS).convert('L')
pil_image2 = Image.open('edges2.jpg').resize((config.img_height, config.img_width), Image.ANTIALIAS).convert('L')
image = np.array(pil_image)
image2 = np.array(pil_image2)

cv2.imwrite("g1.png", image)
cv2.imwrite("g2.png", image2)
"""
