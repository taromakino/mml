import numpy as np
import os
from PIL import ImageEnhance
from torchvision import datasets

def group_data(data):
    grouped = {}
    for img, y in data:
        if y not in grouped:
            grouped[y] = []
        grouped[y].append(img)
    return grouped

def make_data(rng, is_trainval, p_shuffle_u):
    data_x0 = datasets.mnist.MNIST(os.environ["DATA_DPATH"], train=is_trainval, download=True)
    data_x1 = datasets.mnist.FashionMNIST(os.environ["DATA_DPATH"], train=is_trainval, download=True)
    grouped_x0 = group_data(data_x0)
    grouped_x1 = group_data(data_x1)
    x0, x1, y = [], [], []
    for class_idx, x0_class in grouped_x0.items():
        x0 += x0_class
        x1 += grouped_x1[class_idx]
        y += [class_idx] * len(x0_class)
    y = np.array(y, dtype="float32")
    n_classes = int(y.max()) + 1
    brightness_range = np.linspace(0.1, 1, n_classes)
    for i in range(len(y)):
        u_elem = int(y[i])
        if rng.uniform() < p_shuffle_u:
            u_elem = rng.randint(n_classes)
        enhancer0 = ImageEnhance.Brightness(x0[i])
        enhancer1 = ImageEnhance.Brightness(x1[i])
        x0[i] = enhancer0.enhance(brightness_range[u_elem])
        x1[i] = enhancer1.enhance(brightness_range[u_elem])
        x0[i] = np.array(x0[i]) / 255
        x1[i] = np.array(x1[i]) / 255
    x0, x1 = np.array(x0, dtype="float32"), np.array(x1, dtype="float32")
    x0 = x0.reshape((len(x0), -1))
    x1 = x1.reshape((len(x1), -1))
    idxs = np.arange(len(y))
    rng.shuffle(idxs)
    x0, x1, y = x0[idxs], x1[idxs], y[idxs]
    return x0, x1, y