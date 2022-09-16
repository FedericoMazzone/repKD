# Download and preprocess CIFAR-100 image dataset at data/cifar100/

from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm

from data_subset_generation_utils import save_data_subset

SUBSET_SIZE = 300000   # max 1281167

dataset = tfds.load("imagenet_resized/32x32")

train_x = list()
train_y = list()

for counter, x in tqdm(enumerate(dataset["train"])):
    if counter == SUBSET_SIZE:
        break
    train_x.append(x["image"])
    train_y.append(x["label"])

train_x = np.array(train_x)
train_y = np.array(train_y)

assert train_x.shape == (SUBSET_SIZE, 32, 32, 3)
assert train_y.shape == (SUBSET_SIZE, )

train_x = np.reshape(train_x, (-1, 3072))
train_y = np.reshape(train_y, (-1, 1))

train_set = np.hstack((train_x, train_y))

data_dir = Path("data/imagenet/")
save_data_subset(train_set, 50000, data_dir, "train")
save_data_subset(train_set, 100000, data_dir, "train")
save_data_subset(train_set, 200000, data_dir, "train")
save_data_subset(train_set, 300000, data_dir, "train")
