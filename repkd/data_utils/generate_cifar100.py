# Download and preprocess CIFAR-100 image dataset at data/cifar100/

from pathlib import Path

import numpy as np
from tensorflow.keras.datasets.cifar100 import load_data as tf_cifar_load_data

from data_subset_generation_utils import save_data_subset

(train_x, train_y), (test_x, test_y) = tf_cifar_load_data()

assert train_x.shape == (50000, 32, 32, 3)
assert train_y.shape == (50000, 1)
assert test_x.shape == (10000, 32, 32, 3)
assert test_y.shape == (10000, 1)

train_x = np.reshape(train_x, (-1, 3072))
train_y = np.reshape(train_y, (-1, 1))
test_x = np.reshape(test_x, (-1, 3072))
test_y = np.reshape(test_y, (-1, 1))

train_set = np.hstack((train_x, train_y))
test_set = np.hstack((test_x, test_y))

data_dir = Path("data/cifar100/")
save_data_subset(train_set, 2000, data_dir, "train")
save_data_subset(train_set, 20000, data_dir, "train")
save_data_subset(train_set, 50000, data_dir, "train")
save_data_subset(test_set, 10000, data_dir, "test")
