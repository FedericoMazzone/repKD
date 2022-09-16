# Download and preprocess EMNIST letters dataset at data/emnist_letters/

from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

from data_subset_generation_utils import save_data_subset

train_data, test_data = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    as_supervised=True
)

train_data = list(train_data)
train_x = np.array([np.flip(np.rot90(np.array(train_data[i][0]).reshape(
    28, 28), 3), axis=1) for i in range(len(train_data))])
train_y = np.array([np.array(train_data[i][1], dtype=np.uint8)
                   for i in range(len(train_data))])

test_data = list(test_data)
test_x = np.array([np.flip(np.rot90(np.array(test_data[i][0]).reshape(
    28, 28), 3), axis=1) for i in range(len(test_data))])
test_y = np.array([np.array(test_data[i][1], dtype=np.uint8)
                  for i in range(len(test_data))])

assert train_x.shape == (88800, 28, 28)
assert train_y.shape == (88800,)
assert test_x.shape == (14800, 28, 28)
assert test_y.shape == (14800,)

train_y = train_y - 1
test_y = test_y - 1

train_x = np.reshape(train_x, (-1, 784))
train_y = np.reshape(train_y, (-1, 1))
test_x = np.reshape(test_x, (-1, 784))
test_y = np.reshape(test_y, (-1, 1))

train_set = np.hstack((train_x, train_y))
test_set = np.hstack((test_x, test_y))
full_dataset = np.vstack((train_set, test_set))

data_dir = Path("data/emnist_letters/")
data_dir.mkdir(parents=True, exist_ok=True)
file_path = data_dir.joinpath("full_dataset")
np.save(file_path, full_dataset)
