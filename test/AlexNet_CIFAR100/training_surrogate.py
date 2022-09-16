from pathlib import Path

import numpy as np
import tensorflow as tf
from repkd import utils
from repkd.data_utils.load_data import load_data
from repkd.training_utils_alexnet import generate_AlexNet, normalize

original_model_path = Path("./test/AlexNet_CIFAR100/models/alexnet_d20000")

folder = original_model_path.parent
original_model_name = original_model_path.name
original_train_size = utils.get_train_size_by_model_name(original_model_name)
cifar_train_size, cifar_test_size = original_train_size, 10000
imagenet_size = 50000
optimizer = tf.keras.optimizers.Adam()
loss = "categorical_crossentropy"

surrogate_model_path = folder.joinpath(f"surrogate_{original_model_name}")

# utils.disable_cuda()

# Loading data

print(f"Loading CIFAR-100 train set.")

cifar_train_x, cifar_train_y = load_data(
    f"data/cifar100/train_subset_{cifar_train_size}.npy",
    number_of_labels=100,
    normalize=False,
    shape=(-1, 32, 32, 3)
)

cifar_train_x = normalize(
    cifar_train_x,
    [0.4914, 0.4822, 0.4465],
    [0.2023, 0.1994, 0.2010]
)

print(f"Loading CIFAR-100 test set.")

cifar_test_x, cifar_test_y = load_data(
    f"data/cifar100/test_subset_{cifar_test_size}.npy",
    number_of_labels=100,
    normalize=False,
    shape=(-1, 32, 32, 3)
)

cifar_test_x = normalize(
    cifar_test_x,
    [0.4914, 0.4822, 0.4465],
    [0.2023, 0.1994, 0.2010]
)

print(f"Loading ImageNet dataset.")

imagenet_x, _ = load_data(
    f"data/imagenet/train_subset_{imagenet_size}.npy",
    normalize=False,
    shape=(-1, 32, 32, 3)
)

imagenet_x = normalize(
    imagenet_x,
    [0.4914, 0.4822, 0.4465],
    [0.2023, 0.1994, 0.2010]
)

# Load original model
print("Loading original model.")
original_model = tf.keras.models.load_model(original_model_path)
_, train_acc = original_model.evaluate(cifar_train_x, cifar_train_y, verbose=0)
_, test_acc = original_model.evaluate(cifar_test_x, cifar_test_y, verbose=0)
print(f"Original model: {train_acc=}, {test_acc=}")

# Generate surrogate model
# surrogate_model = generate_AlexNet()
surrogate_model = tf.keras.models.load_model(surrogate_model_path)

surrogate_model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=['accuracy']
)

# Label surrogate dataset
fine_tune_x_train = imagenet_x
fine_tune_y_train = original_model.predict(fine_tune_x_train)
# fine_tune_y_train = np.argmax(fine_tune_y_train, axis=1)
# fine_tune_y_train = tf.keras.utils.to_categorical(fine_tune_y_train, 100)

_, train_acc = surrogate_model.evaluate(
    cifar_train_x, cifar_train_y, verbose=0)
_, test_acc = surrogate_model.evaluate(cifar_test_x, cifar_test_y, verbose=0)
print(f"Surrogate model: {train_acc=}, {test_acc=}")

# # Fine-tune surrogate model
# print("\nFine-tuning...\n")

# surrogate_model.fit(
#     fine_tune_x_train, fine_tune_y_train,
#     batch_size=100,
#     epochs=10000,
#     validation_data=(cifar_test_x, cifar_test_y),
#     shuffle=True
# )

# _, train_acc = surrogate_model.evaluate(
#     cifar_train_x, cifar_train_y, verbose=0)
# _, test_acc = surrogate_model.evaluate(cifar_test_x, cifar_test_y, verbose=0)
# print(f"Surrogate model: {train_acc=}, {test_acc=}")


class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, test_x, test_y):
        self.test_x = test_x
        self.test_y = test_y
        self.best_acc = 0.0

    def on_train_batch_end(self, batch, logs=None):
        val_loss, val_acc = self.model.evaluate(
            self.test_x, self.test_y, verbose=0)
        print(f"{batch=} - {val_acc=}")
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            print(f"New best val acc {val_acc}!")
            self.best_weights = self.model.get_weights()


save_best_model = SaveBestModel(cifar_test_x, cifar_test_y)


surrogate_model.fit(
    fine_tune_x_train, fine_tune_y_train,
    batch_size=1000,
    epochs=10000,
    validation_data=(cifar_test_x, cifar_test_y),
    callbacks=[save_best_model],
    shuffle=True,
    verbose=0
)

# Set best weigts
surrogate_model.set_weights(save_best_model.best_weights)

_, train_acc = surrogate_model.evaluate(
    cifar_train_x, cifar_train_y, verbose=0)
_, test_acc = surrogate_model.evaluate(cifar_test_x, cifar_test_y, verbose=0)
print(f"Surrogate model: {train_acc=}, {test_acc=}")

# Save the surrogate model
surrogate_model.save(surrogate_model_path)
