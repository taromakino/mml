import numpy as np
import os
from torchvision import datasets

def color_image(img, is_red):
    assert img.ndim == 2
    dtype = img.dtype
    h, w = img.shape
    img = np.reshape(img, [h, w, 1])
    if is_red:
        img = np.concatenate([img, np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        img = np.concatenate([np.zeros((h, w, 1), dtype=dtype), img, np.zeros((h, w, 1), dtype=dtype)], axis=2)
    return img

def make_data(is_trainval, p_flip_color, sigma):
    data = datasets.mnist.MNIST(os.environ["DATA_DPATH"], train=is_trainval, download=True)
    imgs, scalars, y = [], [], []
    for idx, (img, digit) in enumerate(data):
        img = np.array(img)
        y_elem = 0 if digit < 5 else 1
        u_elem = y_elem
        if np.random.uniform() < p_flip_color:
            u_elem = not u_elem
        imgs.append(color_image(img, u_elem))
        scalars.append(2 * y_elem + u_elem + np.random.normal(0, sigma))
        y.append(y_elem)
    imgs = np.array(imgs, dtype="float32")
    imgs = imgs.reshape((len(imgs), -1))
    return imgs, np.array(scalars, dtype="float32"), np.array(y, dtype="float32")