# Download and preprocess fashion MNIST dataset at data/fashion_mnist/

from pathlib import Path

import numpy as np
from tensorflow.keras.datasets.fashion_mnist import \
    load_data as tf_fashion_mnist_load_data

from data_subset_generation_utils import save_data_subset

(train_x, train_y), (test_x, test_y) = tf_fashion_mnist_load_data()

assert train_x.shape == (60000, 28, 28)
assert train_y.shape == (60000,)
assert test_x.shape == (10000, 28, 28)
assert test_y.shape == (10000,)

train_x = np.reshape(train_x, (-1, 784))
train_y = np.reshape(train_y, (-1, 1))
test_x = np.reshape(test_x, (-1, 784))
test_y = np.reshape(test_y, (-1, 1))

train_set = np.hstack((train_x, train_y))
test_set = np.hstack((test_x, test_y))

data_dir = Path("data/fashion_mnist/")
save_data_subset(train_set, 100, data_dir, "train")
save_data_subset(train_set, 200, data_dir, "train")
save_data_subset(train_set, 400, data_dir, "train")
save_data_subset(train_set, 600, data_dir, "train")
save_data_subset(train_set, 800, data_dir, "train")
save_data_subset(train_set, 1000, data_dir, "train")
save_data_subset(train_set, 5000, data_dir, "train")
save_data_subset(train_set, 10000, data_dir, "train")
save_data_subset(train_set, 60000, data_dir, "train")
save_data_subset(test_set, 1000, data_dir, "test")
save_data_subset(test_set, 10000, data_dir, "test")
