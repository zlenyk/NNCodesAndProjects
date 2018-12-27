import numpy as np
from skimage import transform

def scale_and_rotate_image(image, angle_range=15.0, scale_range=0.1):
    angle = 2 * angle_range * np.random.random() - angle_range
    scale = 1 + 2 * scale_range * np.random.random() - scale_range

    tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(angle))
    tf_scale = transform.SimilarityTransform(scale=scale)
    tf_shift = transform.SimilarityTransform(translation=[-14, -14])
    tf_shift_inv = transform.SimilarityTransform(translation=[14, 14])

    image = transform.warp(image,
                           (tf_shift + tf_scale + tf_rotate + tf_shift_inv).inverse)
    return image

def crop(images, c1, c2):
    images = np.delete(images, range(c1), axis=1)
    images = np.delete(images, range(images.shape[1]-8+c1,images.shape[1]), axis=1)
    images = np.delete(images, range(c2), axis=2)
    images = np.delete(images, range(images.shape[2]-8+c2,images.shape[2]), axis=2)
    return images

def crop_images(images):
    return crop(images, np.random.randint(0,8), np.random.randint(0,8))

def crop_centrally(images):
    return crop(images, 4, 4)
