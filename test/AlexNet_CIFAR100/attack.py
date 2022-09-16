from pathlib import Path

import numpy as np
import tensorflow as tf
from repkd import utils
from repkd.attacks.nasr_2019_whitebox import attack_nasr_2019
from repkd.data_utils.load_data import load_data
from repkd.training_utils_alexnet import normalize
from tensorflow.keras.utils import to_categorical

MODEL_PATH = Path(
    "./test/AlexNet_CIFAR100/models/alexnet_d20000_no_overfit")

TRAIN_SIZE = utils.get_train_size_by_model_name(MODEL_PATH.name)

TRAIN_SET_PATH = f"data/cifar100/train_subset_{TRAIN_SIZE}.npy"
FULL_TRAIN_SET_PATH = "data/cifar100/train_subset_50000.npy"
FULL_TEST_SET_PATH = f"data/cifar100/test_subset_10000.npy"
NUMBER_OF_CLASSES = 100


def main():

    utils.log(f"Attacking model at {MODEL_PATH}")

    # Load model
    target_model = tf.keras.models.load_model(MODEL_PATH)

    # Load data
    train_x, train_y = load_data(
        TRAIN_SET_PATH,
        number_of_labels=None,
        normalize=False,
        shape=(-1, 32, 32, 3)
    )

    full_train_x, full_train_y = load_data(
        FULL_TRAIN_SET_PATH,
        number_of_labels=None,
        normalize=False,
        shape=(-1, 32, 32, 3)
    )

    full_test_x, full_test_y = load_data(
        FULL_TEST_SET_PATH,
        number_of_labels=None,
        normalize=False,
        shape=(-1, 32, 32, 3)
    )

    train_x = normalize(
        train_x,
        [0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.2010]
    )

    full_train_x = normalize(
        full_train_x,
        [0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.2010]
    )

    full_test_x = normalize(
        full_test_x,
        [0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.2010]
    )

    assert np.array_equal(train_x, full_train_x[:TRAIN_SIZE])
    assert np.array_equal(train_y, full_train_y[:TRAIN_SIZE])

    # Testing model
    train_loss, train_acc = target_model.evaluate(
        train_x, to_categorical(train_y, NUMBER_OF_CLASSES), verbose=0)
    test_loss, test_acc = target_model.evaluate(
        full_test_x, to_categorical(full_test_y, NUMBER_OF_CLASSES), verbose=0)
    utils.log(f"Target model {train_acc=}, {test_acc=}")

    # Attack with Nasr 2019 whitebox
    utils.log("\nAttack Nasr 2019 whitebox\n")
    attack_accuracy = attack_nasr_2019.attack(
        target_model,
        train_x, train_y,
        # full_test_x, full_test_y,
        full_train_x[TRAIN_SIZE:], full_train_y[TRAIN_SIZE:],
        num_class=None,
        layers_to_exploit=[11],
        gradients_to_exploit=[],
        exploit_loss=True,
        exploit_label=True,
        epochs=100,
        batch_size=100,
        attack_percentage=0.50,
        number_of_repetitions=1
    )


if __name__ == "__main__":
    main()
