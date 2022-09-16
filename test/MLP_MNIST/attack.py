from pathlib import Path

import numpy as np
import tensorflow as tf
from repkd import utils
from repkd.attacks.nasr_2019_whitebox import attack_nasr_2019
from repkd.attacks.shokri_2017_blackbox import attack_shokri_2017
from repkd.data_utils.load_data import load_data
from repkd.training_utils import generate_MLP, train_model
from tensorflow.keras.utils import to_categorical

# MODEL_PATH = Path("./test/MNIST_MLP/models/d400_e1000/"
#                   "model_d400_oadam_lcc_b10_e1000_a30-10")
# MODEL_PATH = Path("./test/MNIST_MLP/models/d400_e1000/"
#                   "surrogate_model_d400_oadam_lcc_b10_e1000_a30-10")
# MODEL_PATH = Path("./test/MNIST_MLP/models/d400uniform/"
#                   "model_d400_oadam_lcc_b10_e1000_a30-10")
# MODEL_PATH = Path("./test/MNIST_MLP/models/d400uniform/"
#                   "surrogate_model_d400_oadam_lcc_b10_e1000_a30-10")
# MODEL_PATH = Path("./test/MNIST_MLP/models/d100uniform/"
#                   "model0_d100_oadam_lcc_b10_a30-10")
MODEL_PATH = Path("./test/MNIST_MLP/models/d100uniform/"
                  "surrogate_model0_d100_oadam_lcc_b10_a30-10")

# uniform = ""
uniform = "_uniform"

ARCHITECTURE = utils.get_architecture_by_model_name(MODEL_PATH.name)
BATCH_SIZE = utils.get_batch_size_by_model_name(MODEL_PATH.name)
TRAIN_SIZE = utils.get_train_size_by_model_name(MODEL_PATH.name)
ACCURACY_THRESHOLD = 0.95
MAX_EPOCHS = 1000
LOSS = "categorical_crossentropy"
OPTIMIZER = tf.keras.optimizers.Adam()

NUMBER_OF_SHADOW_MODELS = 100
NUM_WORKERS = 10
SHOKRI_ATTACK_MODEL_PATH = MODEL_PATH.parent.joinpath(
    f"attack_model_s{NUMBER_OF_SHADOW_MODELS}")

TRAIN_SET_PATH = f"data/mnist_digits/train{uniform}_subset_{TRAIN_SIZE}.npy"
FULL_TRAIN_SET_PATH = "data/mnist_digits/train_subset_10000.npy"
TEST_SET_PATH = f"data/mnist_digits/test{uniform}_subset_{TRAIN_SIZE}.npy"
FULL_TEST_SET_PATH = f"data/mnist_digits/test_subset_10000.npy"
NUMBER_OF_CLASSES = 10

# These 3 functions MUST be top-level functions, i.e. defined in this level of
# the module, otherwise issues with the pooling.


def shadow_model_generate():
    return generate_MLP(ARCHITECTURE)


def shadow_model_train(shadow_model, shadow_train_x, shadow_train_y):
    train_model(
        shadow_model,
        shadow_train_x,
        tf.keras.utils.to_categorical(shadow_train_y, NUMBER_OF_CLASSES),
        None, None,
        accuracy_threshold=ACCURACY_THRESHOLD,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        loss=LOSS,
        optimizer=OPTIMIZER,
        verbose=0
    )


def model_predict(shadow_model, shadow_train_x):
    return shadow_model.predict(shadow_train_x)


def main():

    utils.log(f"Attacking model at {MODEL_PATH}")

    utils.disable_cuda()

    # Load model
    target_model = tf.keras.models.load_model(MODEL_PATH)

    # Load data
    train_x, train_y = load_data(TRAIN_SET_PATH)
    full_train_x, full_train_y = load_data(FULL_TRAIN_SET_PATH)
    test_x, test_y = load_data(TEST_SET_PATH)
    full_test_x, full_test_y = load_data(FULL_TEST_SET_PATH)

    # assert np.array_equal(train_x, full_train_x[:TRAIN_SIZE])
    # assert np.array_equal(train_y, full_train_y[:TRAIN_SIZE])

    ttrain = [(train_y == i).sum() for i in range(NUMBER_OF_CLASSES)]
    ttrain1 = [(train_y[:int(0.8 * TRAIN_SIZE)] == i).sum()
               for i in range(NUMBER_OF_CLASSES)]
    ttrain2 = [(train_y[int(0.8 * TRAIN_SIZE):TRAIN_SIZE] == i).sum()
               for i in range(NUMBER_OF_CLASSES)]
    ttest = [(test_y[:TRAIN_SIZE] == i).sum()
             for i in range(NUMBER_OF_CLASSES)]

    print(ttrain)
    print(ttrain1)
    print(ttrain2)
    print(ttest)

    # import sys
    # sys.exit()

    # Testing model
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
    #     layers_to_exploit=[2],
    #     gradients_to_exploit=[],
    #     exploit_loss=True,
    #     exploit_label=True,
    #     epochs=100,
    #     batch_size=10,
    #     attack_percentage=0.50,
    #     number_of_repetitions=4
    # )

    # Attack with Shokri 2017 blackbox
    utils.log("\nAttack Shokri 2017 blackbox\n")

    # Train attack models
    attack_model_RFC, attack_model_EST =\
        attack_shokri_2017.train_attack_models(
            shadow_model_generate,
            shadow_model_train,
            model_predict,
            full_train_x[4*TRAIN_SIZE:], full_train_y[4*TRAIN_SIZE:],
            full_test_x[4*TRAIN_SIZE:], full_test_y[4*TRAIN_SIZE:],
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
