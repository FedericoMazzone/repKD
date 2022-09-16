from pathlib import Path

import numpy as np
import tensorflow as tf
from repkd import utils
from repkd.attacks.nasr_2019_whitebox import attack_nasr_2019
from repkd.attacks.shokri_2017_blackbox import attack_shokri_2017
from repkd.data_utils.load_data import load_data
from repkd.training_utils_resnet import generate_ResNet_ws
from tensorflow.keras.utils import to_categorical

MODEL_PATH = Path("./test/ResNet18_Fashion/models/exp_ws/original")

TRAIN_SIZE = 600

NUMBER_OF_SHADOW_MODELS = 10
NUM_WORKERS = 1
SHOKRI_ATTACK_MODEL_PATH = MODEL_PATH.parent.joinpath(
    f"attack_model_s{NUMBER_OF_SHADOW_MODELS}")


TRAIN_SET_PATH = f"data/fashion_mnist/train_subset_{TRAIN_SIZE}.npy"
FULL_TRAIN_SET_PATH = "data/fashion_mnist/train_subset_60000.npy"
FULL_TEST_SET_PATH = "data/fashion_mnist/test_subset_10000.npy"
NUMBER_OF_CLASSES = 10


# These 3 functions MUST be top-level functions, i.e. defined in this level of
# the module, otherwise issues with the pooling.

def shadow_model_generate():
    return generate_ResNet_ws()


def shadow_model_train(shadow_model, shadow_train_x, shadow_train_y):

    shadow_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    shadow_model.fit(
        shadow_train_x,
        tf.keras.utils.to_categorical(shadow_train_y, NUMBER_OF_CLASSES),
        batch_size=100,
        epochs=100,
        shuffle=True,
        # verbose=0
    )


def model_predict(shadow_model, shadow_train_x):
    return shadow_model.predict(shadow_train_x)


def main():

    utils.log(f"Attacking model at {MODEL_PATH}")

    # utils.disable_cuda()

    # Load model
    target_model = tf.keras.models.load_model(MODEL_PATH)

    # Load data
    train_x, train_y = load_data(TRAIN_SET_PATH, shape=(-1, 28, 28, 1))
    full_train_x, full_train_y = load_data(
        FULL_TRAIN_SET_PATH, shape=(-1, 28, 28, 1))
    full_test_x, full_test_y = load_data(
        FULL_TEST_SET_PATH, shape=(-1, 28, 28, 1))
    test_x, test_y = full_test_x[:TRAIN_SIZE], full_test_y[:TRAIN_SIZE]

    assert np.array_equal(train_x, full_train_x[:TRAIN_SIZE])
    assert np.array_equal(train_y, full_train_y[:TRAIN_SIZE])

    # Evaluate the model
    train_loss, train_acc = target_model.evaluate(
        train_x, to_categorical(train_y, NUMBER_OF_CLASSES), verbose=0)
    test_loss, test_acc = target_model.evaluate(
        full_test_x, to_categorical(full_test_y, NUMBER_OF_CLASSES), verbose=0)
    utils.log(f"Target model {train_acc=}, {test_acc=}")

    # # Attack with Nasr 2019 whitebox
    # utils.log("\nAttack Nasr 2019 whitebox\n")
    # attack_accuracy = attack_nasr_2019.attack(
    #     target_model,
    #     train_x, train_y,
    #     test_x, test_y,
    #     num_class=NUMBER_OF_CLASSES,
    #     layers_to_exploit=[len(target_model.layers)],
    #     gradients_to_exploit=[],
    #     exploit_loss=True,
    #     exploit_label=True,
    #     epochs=10,
    #     batch_size=10,
    #     attack_percentage=0.50,
    #     number_of_repetitions=1
    # )

    # Attack with Shokri 2017 blackbox
    utils.log("\nAttack Shokri 2017 blackbox\n")

    # Train attack models
    attack_model_RFC, attack_model_EST = attack_shokri_2017.train_attack_models(
        shadow_model_generate,
        shadow_model_train,
        model_predict,
        full_train_x[TRAIN_SIZE:], full_train_y[TRAIN_SIZE:],
        full_test_x[TRAIN_SIZE:], full_test_y[TRAIN_SIZE:],
        TRAIN_SIZE,
        NUMBER_OF_CLASSES,
        number_of_shadow_models=NUMBER_OF_SHADOW_MODELS,
        num_workers=NUM_WORKERS
    )

    # Save attack models
    attack_model_RFC.save(f"{SHOKRI_ATTACK_MODEL_PATH}.rfc")
    attack_model_EST.save(f"{SHOKRI_ATTACK_MODEL_PATH}.est")

    # Load attack models
    attack_model_RFC = attack_shokri_2017.AttackModel.load(
        f"{SHOKRI_ATTACK_MODEL_PATH}.rfc")
    attack_model_EST = attack_shokri_2017.AttackModel.load(
        f"{SHOKRI_ATTACK_MODEL_PATH}.est")

    # Run attack
    utils.log("Attack with Shokri RFC")
    attack_shokri_2017.attack(
        target_model,
        train_x, train_y,
        test_x, test_y,
        NUMBER_OF_CLASSES,
        attack_model=attack_model_RFC
    )
    utils.log("Attack with Shokri EST")
    attack_shokri_2017.attack(
        target_model,
        train_x, train_y,
        test_x, test_y,
        NUMBER_OF_CLASSES,
        attack_model=attack_model_EST
    )


if __name__ == "__main__":
    main()
