import numpy as np


def to_gryscale(img):
    return np.mean(img, axis=2).astype('uint8')


def downsample(img):
    return img[::2, ::2]


def crop_image(img):
    return img[30:-10, 6:-6]