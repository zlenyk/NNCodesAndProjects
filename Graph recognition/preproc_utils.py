import numpy as np
import cv2
import config

def basic_filter(image):
    image =  \
        ImageEnhance.Contrast(image).enhance(2)     \
        .filter(ImageFilter.UnsharpMask(percent = 200)) \
        .point(lambda p: p * 3)
    show_image('filter', np.array(image))
    return image

def closing_wrapper(image):
    kernel = np.ones((3,3),np.uint8)
    image = abs(255-image)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
    image = cv2.dilate(image, kernel, iterations=0)

    image = abs(255-image)
    return image

def threshold(image):
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=2*int(config.threshold_block_size_ratio*config.img_width / 2)+5,
        C=config.threshold_mean,
    )
    return image

def noise_cleaning(image):
    copy = np.array(image)
    _,contours,_ = cv2.findContours(copy, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config.img_width*config.img_height * config.noise_ratio:
            cv2.drawContours(image, [cnt], 0, color=255, thickness=-1)

    # the same thing with filling in black (for small white holes)
    copy = np.array(image)
    _,contours,_ = cv2.findContours(copy, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config.img_width*config.img_height * config.noise_ratio:
            cv2.drawContours(image, [cnt], 0, color=0, thickness=-1)

    return image

# for parsing bool arguments
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
