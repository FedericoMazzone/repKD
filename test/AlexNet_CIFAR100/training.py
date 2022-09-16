import tensorflow as tf
from repkd.data_utils.load_data import load_data
from repkd.training_utils_alexnet import generate_AlexNet, normalize

train_x, train_y = load_data(
    "data/cifar100/train_subset_20000.npy",
    number_of_labels=100,
    normalize=False,
    shape=(-1, 32, 32, 3)
)

test_x, test_y = load_data(
    "data/cifar100/test_subset_10000.npy",
    number_of_labels=100,
    normalize=False,
    shape=(-1, 32, 32, 3)
)

train_x = normalize(
    train_x,
    [0.4914, 0.4822, 0.4465],
    [0.2023, 0.1994, 0.2010]
)

test_x = normalize(
    test_x,
    [0.4914, 0.4822, 0.4465],
    [0.2023, 0.1994, 0.2010]
)

model = generate_AlexNet()

# callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

model.fit(
    train_x, train_y,
    batch_size=128,
    epochs=10000,
    validation_data=(test_x, test_y),
    shuffle=True
    # callbacks=[callback]
)

_, train_acc = model.evaluate(train_x, train_y, verbose=0)
_, test_acc = model.evaluate(test_x, test_y, verbose=0)
print(f"Surrogate model: {train_acc=}, {test_acc=}")

# Save the surrogate model
model.save("./test/AlexNet_CIFAR100/models/alexnet_d20000_no_overfit")
