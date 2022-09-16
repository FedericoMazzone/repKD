from pathlib import Path

import numpy as np
import tensorflow as tf
from repkd import utils
from repkd.data_utils.load_data import load_data
from repkd.kd import kd
from repkd.training_utils_resnet import generate_ResNet

MODEL_PATH = Path("./test/ResNet18_Fashion/models/original_model_d600")

TRAIN_SIZE = utils.get_train_size_by_model_name(MODEL_PATH.name)

TRAIN_SET_PATH = "data/fashion_mnist/train_subset_600.npy"
FULL_TRAIN_SET_PATH = "data/fashion_mnist/train_subset_60000.npy"
FULL_TEST_SET_PATH = "data/fashion_mnist/test_subset_10000.npy"
NUM_CLASSES = 10

# Auxiliary functions for surrogate training


def generate_model():
    return generate_ResNet()


def noise_perturbation(prediction, noise_dev):
    noise = np.random.normal(0, noise_dev, prediction.shape)
    return prediction + noise


def label_only_masking(prediction):
    label = np.argmax(prediction, axis=1)
    return tf.keras.utils.to_categorical(label, NUM_CLASSES)


# Load model
target_model = tf.keras.models.load_model(MODEL_PATH)

# Load data
train_x, train_y = load_data(
    TRAIN_SET_PATH,
    number_of_labels=10,
    shape=(-1, 28, 28, 1)
)
full_train_x, full_train_y = load_data(
    FULL_TRAIN_SET_PATH,
    number_of_labels=10,
    shape=(-1, 28, 28, 1)
)
test_x, test_y = load_data(
    FULL_TEST_SET_PATH,
    number_of_labels=10,
    shape=(-1, 28, 28, 1)
)

# Evaluate the model
train_loss, train_acc = target_model.evaluate(train_x, train_y, verbose=0)
test_loss, test_acc = target_model.evaluate(test_x, test_y, verbose=0)
utils.log(f"Target model {train_acc=}, {test_acc=}")

# Train surrogate model
surrogate_model = kd(
    teacher_model=target_model,
    original_train_x=train_x,
    original_train_y=train_y,
    original_test_x=test_x,
    original_test_y=test_y,
    surrogate_data_x=full_train_x[10000:],
    generate_model=generate_model,
    generate_autoencoder=None,
    original_val_x=full_train_x[TRAIN_SIZE:10000],
    original_val_y=full_train_y[TRAIN_SIZE:10000],
    # conf_masking=None,
    conf_masking=lambda p: noise_perturbation(p, 0.001),
    ft_loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    ft_optimizer=tf.keras.optimizers.Adam(),
    ft_batch=100,
    ft_epochs=2,
    bm_batch=1000,
    bm_epochs=1
)

# Save the surrogate model
surrogate_model_path = MODEL_PATH.parent.joinpath(
    f"surrogate_{MODEL_PATH.name}")
surrogate_model.save(surrogate_model_path)
