from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython import embed
from repkd import utils
from repkd.data_utils.load_data import load_data
from repkd.training_utils import generate_MLP

utils.disable_cuda()

original_model_path = Path(
    "./test/MNIST_MLP/models/d100uniform/model0_d100_oadam_lcc_b10_a30-10")
# uniform = ""
uniform = "_uniform"
folder = original_model_path.parent
original_model_name = original_model_path.name
original_train_size = utils.get_train_size_by_model_name(original_model_name)
architecture = utils.get_architecture_by_model_name(original_model_name)
letters_train_size, letters_test_size = 71040, 14800
digits_train_size, digits_test_size = original_train_size, 10000
accuracy_threshold = 0.86
max_epochs = 10000
batch_size = 10
optimizer = tf.keras.optimizers.Adam()
loss = "categorical_crossentropy"

# Load EMNIST letters data for training the surrogate model
letters_train_path = \
    f"data/emnist_letters/train_subset_{letters_train_size}.npy"
letters_test_path = \
    f"data/emnist_letters/test_subset_{letters_test_size}.npy"
x_train, y_train = load_data(letters_train_path, 26)
x_test, y_test = load_data(letters_test_path, 26)

# Load MNIST digits data for testing the surrogate model
digits_train_path = f"data/mnist_digits/train{uniform}_subset_{digits_train_size}.npy"
digits_test_path = f"data/mnist_digits/test_subset_{digits_test_size}.npy"
print(f"Loading MNIST train data from {digits_train_path}")
mnist_x_train, mnist_y_train = load_data(digits_train_path, 10)
print(f"Loading MNIST test data from {digits_test_path}")
mnist_x_test, mnist_y_test = load_data(digits_test_path, 10)

# Load original model
original_model = tf.keras.models.load_model(original_model_path)
print("Original model accuracy on MNIST train set:")
original_model.evaluate(mnist_x_train, mnist_y_train)
print("Original model accuracy on MNIST test set:")
original_model.evaluate(mnist_x_test, mnist_y_test)

# Generate surrogate model
surrogate_model = generate_MLP(architecture)

# Generate autoencoder
autoencoder_architecture = architecture + architecture[-2::-1] + [784]
autoencoder = generate_MLP(autoencoder_architecture)

# Pre-train surrogate model
print("\nPre-training...\n")

autoencoder.compile(
    loss="mean_squared_error",
    optimizer=tf.keras.optimizers.Adam()
)
autoencoder.fit(
    x_train, x_train,
    batch_size=100,
    epochs=100,
    shuffle=True
)


def show():
    decoded_imgs = autoencoder.predict(x_test)
    n = 5
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# Copying encoder into surrogate model
for i in range(len(surrogate_model.layers)):
    surrogate_model.layers[i].set_weights(autoencoder.layers[i].get_weights())

surrogate_model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=['accuracy']
)

# Label letters as digits
fine_tune_x_train = x_train
# fine_tune_x_train = x_train_reduced
# fine_tune_x_train = x_train_reduced2
fine_tune_y_train = original_model.predict(fine_tune_x_train)
# print(fine_tune_y_train[:16])
# print(np.argmax(fine_tune_y_train[:16], axis=1))
# fine_tune_y_train = (fine_tune_y_train + np.random.normal(0, 0.0005, fine_tune_y_train.shape)).clip(0, 1)
fine_tune_y_train = tf.keras.utils.to_categorical(
    np.argmax(fine_tune_y_train, axis=1), 10)


# def softmax(x, t=1):
#     # return np.exp(x / t) / np.sum(np.exp(x / t), axis=1)[:,np.newaxis]
#     e_x = np.exp((x - np.max(x, axis=1)[:, np.newaxis]) / t)
#     return e_x / np.sum(e_x, axis=1)[:, np.newaxis]
# fine_tune_y_train = softmax(fine_tune_y_train, t=0.00001)

# # Freezing first layer of surrogate model
# surrogate_model.layers[0].trainable = False


# Fine-tune surrogate model
print("\nFine-tuning...\n")

# surrogate_model.fit(
#     fine_tune_x_train, fine_tune_y_train,
#     batch_size=10,
#     epochs=1,
#     validation_data=(mnist_x_test, mnist_y_test)
# )


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


save_best_model = SaveBestModel(mnist_x_test, mnist_y_test)

surrogate_model.fit(
    fine_tune_x_train, fine_tune_y_train,
    batch_size=10,
    epochs=5,
    validation_data=(mnist_x_test, mnist_y_test)
)

surrogate_model.fit(
    fine_tune_x_train, fine_tune_y_train,
    batch_size=100,
    epochs=1,
    validation_data=(mnist_x_test, mnist_y_test),
    callbacks=[save_best_model],
    verbose=0
)

# Set best weigts
surrogate_model.set_weights(save_best_model.best_weights)

# Test surrogate_model on MNIST train and test sets
print("Surrogate model accuracy on MNIST train set:")
surrogate_model.evaluate(mnist_x_train, mnist_y_train)
print("Surrogate model accuracy on MNIST test set:")
surrogate_model.evaluate(mnist_x_test, mnist_y_test)

# Save the surrogate model
surrogate_model_path = folder.joinpath(f"surrogate_{original_model_name}")
surrogate_model.save(surrogate_model_path)


fine_tune_y_train = original_model.predict(fine_tune_x_train)

surrogate_model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=['accuracy']
)

surrogate_model.fit(
    fine_tune_x_train, fine_tune_y_train,
    batch_size=100,
    epochs=1,
    validation_data=(mnist_x_test, mnist_y_test),
    callbacks=[save_best_model],
    verbose=0
)

# Set best weigts
surrogate_model.set_weights(save_best_model.best_weights)

# Test surrogate_model on MNIST train and test sets
print("Surrogate model accuracy on MNIST train set:")
surrogate_model.evaluate(mnist_x_train, mnist_y_train)
print("Surrogate model accuracy on MNIST test set:")
surrogate_model.evaluate(mnist_x_test, mnist_y_test)

# Save the surrogate model
surrogate_model_path = folder.joinpath(f"surrogate_fine_{original_model_name}")
surrogate_model.save(surrogate_model_path)
