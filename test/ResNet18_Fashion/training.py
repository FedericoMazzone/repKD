from pathlib import Path

import tensorflow as tf
from repkd.data_utils.load_data import load_data
from repkd.training_utils_resnet import generate_ResNet

train_x, train_y = load_data(
    "data/fashion_mnist/train_subset_600.npy",
    number_of_labels=10,
    normalize=True,
    shape=(-1, 28, 28, 1)
)

test_x, test_y = load_data(
    "data/fashion_mnist/test_subset_10000.npy",
    number_of_labels=10,
    normalize=True,
    shape=(-1, 28, 28, 1)
)

model = generate_ResNet()

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

model.fit(
    train_x, train_y,
    batch_size=100,
    epochs=100,
    # validation_data=(test_x[:600], test_y[:600]),
    shuffle=True
)

# Evaluate the model
_, train_acc = model.evaluate(train_x, train_y, verbose=0)
_, test_acc = model.evaluate(test_x, test_y, verbose=0)
print(f"Model accuracy: train={train_acc}, test={test_acc}")

FOLDER = Path("./test/Fashion_ResNet18/models/")
MODEL_PATH = FOLDER.joinpath("original_model_d600")

# Save the model
FOLDER.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")
