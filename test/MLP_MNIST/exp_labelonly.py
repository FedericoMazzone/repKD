import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from repkd import utils
from repkd.attacks.nasr_2019_whitebox import attack_nasr_2019
from repkd.attacks.shokri_2017_blackbox import attack_shokri_2017
from repkd.data_utils.load_data import load_data
from repkd.kd import kd
from repkd.training_utils import generate_MLP, train_model
from tensorflow.keras.utils import to_categorical

FOLDER = Path("./test/MNIST_MLP/models/exp_labelonly")
ARCHITECTURE = [30, 10]
TRAIN_SIZE = 400
OPTIMIZER = tf.keras.optimizers.Adam()
LOSS = "categorical_crossentropy"
TRAIN_ACCURACY_THRESHOLD = 1.1
TRAIN_MAX_EPOCHS = 100
TRAIN_BATCH_SIZE = 10
NUM_CLASSES = 10
SHOKRI_ATK_NUM_SHADOW_MODELS = 100
SHOKRI_ATK_NUM_WORKERS = 10
NASR_ATK_LAYERS = [len(ARCHITECTURE)]
NASR_ATK_GRADIENTS = []
NASR_ATK_LOSS = True
NASR_ATK_LABEL = True
NASR_ATK_EPOCHS = 100
NASR_ATK_BATCH = 10
NASR_ATK_PERCENTAGE = 0.50
NASR_ATK_REPETITION = 4


# Auxiliary functions for Shokri attack

def shadow_model_generate():
    return generate_MLP(ARCHITECTURE)


def shadow_model_train(shadow_model, shadow_train_x, shadow_train_y):
    train_model(
        shadow_model,
        shadow_train_x,
        tf.keras.utils.to_categorical(shadow_train_y, NUM_CLASSES),
        None, None,
        accuracy_threshold=TRAIN_ACCURACY_THRESHOLD,
        max_epochs=TRAIN_MAX_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        loss=LOSS,
        optimizer=OPTIMIZER,
        verbose=0
    )


def model_predict(shadow_model, shadow_train_x):
    return shadow_model.predict(shadow_train_x)


# Auxiliary functions for surrogate training

def generate_model():
    return generate_MLP(ARCHITECTURE)


def generate_autoencoder():
    autoencoder_architecture = ARCHITECTURE + ARCHITECTURE[-2::-1] + [784]
    return generate_MLP(autoencoder_architecture)


# def noise_perturbation(prediction, noise_dev):
#     noise = np.random.normal(0, noise_dev, prediction.shape)
#     return (prediction + noise).clip(0, 1)


def label_only_masking(prediction):
    label = np.argmax(prediction, axis=1)
    return tf.keras.utils.to_categorical(label, NUM_CLASSES)


def main():

    MODEL_NAME = f"model{os.getpid()}_d{TRAIN_SIZE}"
    MODEL_PATH = FOLDER.joinpath(MODEL_NAME)
    DIGITS_TRAIN_PATH = f"data/mnist_digits/train_uniform_subset_{TRAIN_SIZE}.npy"
    DIGITS_FULL_TRAIN_PATH = "data/mnist_digits/train_subset_60000.npy"
    DIGITS_TEST_PATH = f"data/mnist_digits/test_uniform_subset_{TRAIN_SIZE}.npy"
    DIGITS_FULL_TEST_PATH = "data/mnist_digits/test_subset_10000.npy"
    LETTERS_TRAIN_PATH = "data/emnist_letters/full_dataset.npy"
    SHOKRI_ATK_MODEL_PATH = "./test/MNIST_MLP/models/exp_nomask/attack_model"

    # Disable cuda
    utils.disable_cuda()

    # Load MNIST digits data
    utils.log(f"Loading MNIST train data from {DIGITS_TRAIN_PATH}")
    mnist_train_x, mnist_train_y = load_data(DIGITS_TRAIN_PATH)
    utils.log(f"Loading MNIST full train data from {DIGITS_FULL_TRAIN_PATH}")
    mnist_full_train_x, mnist_full_train_y = load_data(DIGITS_FULL_TRAIN_PATH)
    utils.log(f"Loading MNIST test data from {DIGITS_TEST_PATH}")
    mnist_test_x, mnist_test_y = load_data(DIGITS_TEST_PATH)
    utils.log(f"Loading MNIST full test data from {DIGITS_FULL_TEST_PATH}")
    mnist_full_test_x, mnist_full_test_y = load_data(DIGITS_FULL_TEST_PATH)

    mnist_train_y_cat = to_categorical(mnist_train_y)
    mnist_full_train_y_cat = to_categorical(mnist_full_train_y)
    mnist_test_y_cat = to_categorical(mnist_test_y)
    mnist_full_test_y_cat = to_categorical(mnist_full_test_y)

    # Load EMNIST letters data
    utils.log(f"Loading EMNIST letters data from {LETTERS_TRAIN_PATH}")
    letters_x, _ = load_data(LETTERS_TRAIN_PATH)

    # # Train attack models
    # attack_model_RFC, attack_model_EST =\
    #     attack_shokri_2017.train_attack_models(
    #         shadow_model_generate,
    #         shadow_model_train,
    #         model_predict,
    #         mnist_full_train_x[4 *
    #                            TRAIN_SIZE:], mnist_full_train_y[4*TRAIN_SIZE:],
    #         mnist_full_test_x[4*TRAIN_SIZE:], mnist_full_test_y[4*TRAIN_SIZE:],
    #         TRAIN_SIZE,
    #         NUM_CLASSES,
    #         number_of_shadow_models=SHOKRI_ATK_NUM_SHADOW_MODELS,
    #         num_workers=SHOKRI_ATK_NUM_WORKERS
    #     )

    # # Save attack models
    # attack_model_RFC.save(f"{SHOKRI_ATK_MODEL_PATH}.rfc")
    # attack_model_EST.save(f"{SHOKRI_ATK_MODEL_PATH}.est")

    # Load attack models
    attack_model_RFC = attack_shokri_2017.AttackModel.load(
        f"{SHOKRI_ATK_MODEL_PATH}.rfc")
    attack_model_EST = attack_shokri_2017.AttackModel.load(
        f"{SHOKRI_ATK_MODEL_PATH}.est")

    def attack(target_model):
        utils.log("Attack with Shokri RFC")
        attack_shokri_2017.attack(
            target_model,
            mnist_train_x, mnist_train_y,
            mnist_test_x, mnist_test_y,
            NUM_CLASSES,
            attack_model=attack_model_RFC
        )
        utils.log("Attack with Shokri EST")
        attack_shokri_2017.attack(
            target_model,
            mnist_train_x, mnist_train_y,
            mnist_test_x, mnist_test_y,
            NUM_CLASSES,
            attack_model=attack_model_EST
        )
        utils.log("Attack with Nasr")
        attack_nasr_2019.attack(
            target_model,
            mnist_train_x, mnist_train_y,
            mnist_test_x, mnist_test_y,
            num_class=NUM_CLASSES,
            layers_to_exploit=NASR_ATK_LAYERS,
            gradients_to_exploit=NASR_ATK_GRADIENTS,
            exploit_loss=NASR_ATK_LOSS,
            exploit_label=NASR_ATK_LABEL,
            epochs=NASR_ATK_EPOCHS,
            batch_size=NASR_ATK_BATCH,
            attack_percentage=NASR_ATK_PERCENTAGE,
            number_of_repetitions=NASR_ATK_REPETITION
        )

    # Create the model
    model = generate_MLP(ARCHITECTURE, model_name=MODEL_NAME)

    # Train the model
    utils.log("Training model")
    train_model(
        model,
        mnist_train_x, mnist_train_y_cat,
        None, None,
        accuracy_threshold=TRAIN_ACCURACY_THRESHOLD,
        max_epochs=TRAIN_MAX_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        loss=LOSS,
        optimizer=OPTIMIZER
    )

    # Evaluate the model
    _, train_acc = model.evaluate(
        mnist_train_x, mnist_train_y_cat, verbose=0)
    _, test_acc = model.evaluate(
        mnist_full_test_x, mnist_full_test_y_cat, verbose=0)
    utils.log(f"Model accuracy: train={train_acc}, test={test_acc}")

    # Save the model
    FOLDER.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    utils.log(f"Model saved at {MODEL_PATH}")

    # Attack the model
    utils.log("Attacking original model")
    attack(model)

    # Train surrogate model
    surrogate_model = kd(
        teacher_model=model,
        original_train_x=mnist_train_x,
        original_train_y=mnist_train_y_cat,
        original_test_x=mnist_test_x,
        original_test_y=mnist_test_y_cat,
        surrogate_data_x=letters_x,
        generate_model=generate_model,
        generate_autoencoder=generate_autoencoder,
        original_val_x=mnist_full_train_x[-10000:],
        original_val_y=mnist_full_train_y_cat[-10000:],
        conf_masking=label_only_masking,
        pt_loss="mean_squared_error",
        pt_optimizer=tf.keras.optimizers.Adam(),
        pt_batch=100,
        pt_epochs=10,
        ft_loss="categorical_crossentropy",
        ft_optimizer=tf.keras.optimizers.Adam(),
        ft_batch=10,
        ft_epochs=2
    )

    # Save the surrogate model
    surrogate_model_path = FOLDER.joinpath(f"surrogate_{MODEL_NAME}")
    surrogate_model.save(surrogate_model_path)

    # Attack surrogate model
    utils.log("Attacking surrogate model")
    attack(surrogate_model)


if __name__ == "__main__":
    main()
