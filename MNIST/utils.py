from skimage import transform
import numpy as np

def scale_and_rotate_image(image, image_shape, angle_range=15.0, scale_range=0.1):
    angle = 2 * angle_range * np.random.random() - angle_range
    scale = 1 + 2 * scale_range * np.random.random() - scale_range

    tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(angle))
    tf_scale = transform.SimilarityTransform(scale=scale)
    tf_shift = transform.SimilarityTransform(translation=[-14, -14])
    tf_shift_inv = transform.SimilarityTransform(translation=[14, 14])

    image = transform.warp(image.reshape(image_shape),
                           (tf_shift + tf_scale + tf_rotate + tf_shift_inv).inverse).ravel()
    return image

def transform_batch(batch, shape):
    return np.apply_along_axis(scale_and_rotate_image, 1, batch, shape)
