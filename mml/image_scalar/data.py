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

def make_data(dataset_name, rng, is_trainval, p_shuffle_u, sigma):
    if dataset_name == "MNIST":
        data = datasets.mnist.MNIST(os.environ["DATA_DPATH"], train=is_trainval, download=True)
    elif dataset_name == "FashionMNIST":
        data = datasets.mnist.FashionMNIST(os.environ["DATA_DPATH"], train=is_trainval, download=True)
    else:
        raise ValueError
    imgs, scalars, y = [], [], []
    for img, digit in data:
        img = np.array(img) / 255
        y_elem = 0 if digit < 5 else 1
        u_elem = y_elem
        if rng.uniform() < p_shuffle_u:
            u_elem = rng.randint(2)
        imgs.append(color_image(img, u_elem))
        scalars.append(2 * y_elem + u_elem + rng.normal(0, sigma))
        y.append(y_elem)
    imgs = np.array(imgs, dtype="float32")
    imgs = imgs.reshape((len(imgs), -1))
    return imgs, np.array(scalars, dtype="float32"), np.array(y, dtype="float32")