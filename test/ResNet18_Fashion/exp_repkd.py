import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from repkd import utils
from repkd.attacks.nasr_2019_whitebox import attack_nasr_2019
from repkd.attacks.shokri_2017_blackbox import attack_shokri_2017
from repkd.data_utils.load_data import load_data
from repkd.kd import kd
from repkd.training_utils_resnet import generate_ResNet_ws
from tensorflow.keras.utils import to_categorical

MODEL_PATH = Path("test/ResNet18_Fashion/models/exp_repkd/original")
FOLDER = MODEL_PATH.parent
TRAIN_SIZE = 600
OPTIMIZER = tf.keras.optimizers.Adam()
LOSS = tf.keras.losses.CategoricalCrossentropy()
TRAIN_ACCURACY_THRESHOLD = 1.1
TRAIN_MAX_EPOCHS = 100
TRAIN_BATCH_SIZE = 100
NUM_CLASSES = 10
SHOKRI_ATK_NUM_SHADOW_MODELS = 10
SHOKRI_ATK_NUM_WORKERS = 1
NASR_ATK_LAYERS = [88]
NASR_ATK_GRADIENTS = []
NASR_ATK_LOSS = True
NASR_ATK_LABEL = True
NASR_ATK_EPOCHS = 50
NASR_ATK_BATCH = 100
NASR_ATK_PERCENTAGE = 0.50
NASR_ATK_REPETITION = 2
NUM_REPKD = 4

# Auxiliary functions for Shokri attack


def shadow_model_generate():
    return generate_ResNet_ws()


def shadow_model_train(shadow_model, shadow_train_x, shadow_train_y):

    shadow_model.compile(
        loss=LOSS,
        optimizer=OPTIMIZER,
        metrics=['accuracy']
    )

    shadow_model.fit(
        shadow_train_x,
        tf.keras.utils.to_categorical(shadow_train_y, NUM_CLASSES),
        batch_size=100,
        epochs=100,
        shuffle=True,
        verbose=0
    )


def model_predict(shadow_model, shadow_train_x):
    return shadow_model.predict(shadow_train_x)


# Auxiliary functions for surrogate training

def generate_model():
    return generate_ResNet_ws()


def noise_perturbation(prediction, noise_dev):
    noise = np.random.normal(0, noise_dev, prediction.shape)
    return (prediction + noise).clip(0, 1)


def label_only_masking(prediction):
    label = np.argmax(prediction, axis=1)
    return tf.keras.utils.to_categorical(label, NUM_CLASSES)


def main():

    TRAIN_SET_PATH = f"data/fashion_mnist/train_subset_{TRAIN_SIZE}.npy"
    FULL_TRAIN_SET_PATH = "data/fashion_mnist/train_subset_60000.npy"
    FULL_TEST_SET_PATH = "data/fashion_mnist/test_subset_10000.npy"
    SHOKRI_ATK_MODEL_PATH = FOLDER.joinpath("attack_model")

    # Define masking function
    if sys.argv[1] == "no_masking":
        utils.log("No masking")
        conf_masking = None
        conf_masking_symbol = "NM"
    elif sys.argv[1] == "label_only":
        utils.log("Label-only masking")
        conf_masking = label_only_masking
        conf_masking_symbol = "LO"
    elif sys.argv[1].startswith("noise=="):
        NOISE_DEV = float(sys.argv[1][7:])
        utils.log(f"Prediction-vector perturbation, noise = {NOISE_DEV}")
        def conf_masking(p): return noise_perturbation(p, NOISE_DEV)
        conf_masking_symbol = f"noise{sys.argv[1][9:]}"

    # Define surrogate data distributor
    if sys.argv[2] == "surr_same":
        utils.log("Full surrogate dataset used every iteration")
        surr_data_distr_symbol = "SD"

        def surr_data_distr(surr_dataset, iteration):
            return surr_dataset
    elif sys.argv[2] == "surr_alternate":
        utils.log("Alternating half surrogate dataset used each iteration")
        surr_data_distr_symbol = "AD"

        def surr_data_distr(surr_dataset, iteration):
            n = len(surr_dataset)
            m = n // 2
            if iteration % 2 == 0:
                return surr_dataset[:m]
            else:
                return surr_dataset[m:]

    # Disable cuda
    # utils.disable_cuda()

    # Load data
    utils.log(f"Loading train data from {TRAIN_SET_PATH}")
    train_x, train_y = load_data(
        TRAIN_SET_PATH,
        shape=(-1, 28, 28, 1)
    )
    utils.log(f"Loading full train data from {FULL_TRAIN_SET_PATH}")
    full_train_x, full_train_y = load_data(
        FULL_TRAIN_SET_PATH,
        shape=(-1, 28, 28, 1)
    )
    utils.log(f"Loading MNIST full test data from {FULL_TEST_SET_PATH}")
    full_test_x, full_test_y = load_data(
        FULL_TEST_SET_PATH,
        shape=(-1, 28, 28, 1)
    )

    train_y_cat = to_categorical(train_y)
    full_train_y_cat = to_categorical(full_train_y)
    full_test_y_cat = to_categorical(full_test_y)

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
        atk_acc_shokriRFC = attack_shokri_2017.attack(
            target_model,
            train_x, train_y,
            full_test_x[:TRAIN_SIZE], full_test_y[:TRAIN_SIZE],
            NUM_CLASSES,
            attack_model=attack_model_RFC
        )
        utils.log("Attack with Shokri EST")
        atk_acc_shokriEST = attack_shokri_2017.attack(
            target_model,
            train_x, train_y,
            full_test_x[:TRAIN_SIZE], full_test_y[:TRAIN_SIZE],
            NUM_CLASSES,
            attack_model=attack_model_EST
        )
        utils.log("Attack with Nasr")
        atk_acc_nasr = attack_nasr_2019.attack(
            target_model,
            train_x, train_y,
            full_test_x, full_test_y,
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
        return atk_acc_shokriRFC, atk_acc_shokriEST, atk_acc_nasr

    # # Create the model
    # model = generate_MLP(ARCHITECTURE, model_name=MODEL_NAME)

    # # Train the model
    # utils.log("Training model")
    # train_model(
    #     model,
    #     mnist_train_x, mnist_train_y_cat,
    #     None, None,
    #     accuracy_threshold=TRAIN_ACCURACY_THRESHOLD,
    #     max_epochs=TRAIN_MAX_EPOCHS,
    #     batch_size=TRAIN_BATCH_SIZE,
    #     loss=LOSS,
    #     optimizer=OPTIMIZER
    # )

    results_summary = f"Results summary for {surr_data_distr_symbol}_{conf_masking_symbol}\n"

    utils.log("\n------\nORIGINAL MODEL\n------\n")

    # Load the mode
    model = tf.keras.models.load_model(MODEL_PATH)

    # Evaluate the model
    _, train_acc = model.evaluate(
        train_x, train_y_cat, verbose=0)
    _, test_acc = model.evaluate(
        full_test_x, full_test_y_cat, verbose=0)
    utils.log(f"Model accuracy: train={train_acc}, test={test_acc}")

    results_summary += f"Original model acc: {train_acc=}, {test_acc=}\n"

    # # Save the model
    # FOLDER.mkdir(parents=True, exist_ok=True)
    # model.save(MODEL_PATH)
    # utils.log(f"Model saved at {MODEL_PATH}")

    # Attack the model
    utils.log("Attacking original model")
    atk_acc_shokriRFC, atk_acc_shokriEST, atk_acc_nasr = attack(model)

    results_summary += f"Original model attack acc: {atk_acc_shokriRFC=}, {atk_acc_shokriEST=}, {atk_acc_nasr=}\n"

    surrogate_model = model

    for i in range(NUM_REPKD):

        utils.log(f"\n------\nSURROGATE MODEL NUMBER {i}\n------\n")

        # Train surrogate model
        surrogate_model = kd(
            teacher_model=surrogate_model,
            original_train_x=train_x,
            original_train_y=train_y_cat,
            original_test_x=full_test_x,
            original_test_y=full_test_y_cat,
            surrogate_data_x=surr_data_distr(full_train_x[10000:], i),
            generate_model=generate_model,
            generate_autoencoder=None,
            original_val_x=full_train_x[TRAIN_SIZE:3*TRAIN_SIZE],
            original_val_y=full_train_y_cat[TRAIN_SIZE:3*TRAIN_SIZE],
            conf_masking=conf_masking,
            ft_loss=LOSS,
            ft_optimizer=OPTIMIZER,
            ft_batch=TRAIN_BATCH_SIZE,
            ft_epochs=1,
            bm_batch=100,
            bm_epochs=1
        )

        # Save the surrogate model
        surrogate_model_name = f"surrogate_{surr_data_distr_symbol}_{conf_masking_symbol}_{i}"
        surrogate_model_path = FOLDER.joinpath(surrogate_model_name)
        surrogate_model.save(surrogate_model_path)

        # Evaluate the surrogate model
        _, surr_train_acc = surrogate_model.evaluate(
            train_x, train_y_cat, verbose=0)
        _, surr_test_acc = surrogate_model.evaluate(
            full_test_x, full_test_y_cat, verbose=0)
        utils.log(
            f"Surrogate model accuracy: train={surr_train_acc}, test={surr_test_acc}")

        results_summary += f"Surrogate model {i} acc: {surr_train_acc=}, {surr_test_acc=}\n"

        # Attack surrogate model
        utils.log(f"Attacking surrogate model {i}")
        atk_acc_shokriRFC, atk_acc_shokriEST, atk_acc_nasr = attack(
            surrogate_model)

        results_summary += f"Surrogate model {i} attack acc: {atk_acc_shokriRFC=}, {atk_acc_shokriEST=}, {atk_acc_nasr=}\n"

    utils.log(results_summary)


if __name__ == "__main__":
    main()
