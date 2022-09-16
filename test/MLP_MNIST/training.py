from pathlib import Path

import tensorflow as tf
from repkd.data_utils.load_data import load_data
from repkd.training_utils import generate_MLP, train_model

architecture = [30, 10]
train_subset_size = 100
# uniform = ""
uniform = "_uniform"
test_subset_size = 10000
optimizer = "adam"
loss = "cc"
accuracy_threshold = 0.95
max_epochs = 1000
batch_size = 10
folder = Path("./test/MNIST_MLP/models/d100uniform")
number_of_models = 10

optimizer_list = {
    "sgd": tf.keras.optimizers.SGD(learning_rate=3.0),
    "adam": tf.keras.optimizers.Adam()
}

loss_list = {
    "mse": "mean_squared_error",
    "cc": "categorical_crossentropy"
}

# Load the datasets
train_set_path = f"data/mnist_digits/train{uniform}_subset_{train_subset_size}.npy"
test_set_path = f"data/mnist_digits/test_subset_{test_subset_size}.npy"
print(f"Loading train data from {train_set_path}")
train_x, train_y = load_data(train_set_path, 10)
print(f"Loading test data from {test_set_path}")
test_x, test_y = load_data(test_set_path, 10)

for i in range(number_of_models):

    # Create the model
    model_name = f"model{i}_d{len(train_x)}_o{optimizer}_l{loss}" + \
        f"_b{batch_size}_a{'-'.join(str(x) for x in architecture)}"
    model = generate_MLP(architecture, model_name=model_name)

    # Train the model
    train_model(
        model,
        train_x, train_y,
        None, None,
        # test_x, test_y,
        accuracy_threshold=accuracy_threshold,
        max_epochs=max_epochs,
        batch_size=batch_size,
        loss=loss_list[loss],
        optimizer=optimizer_list[optimizer]
    )

    # Evaluate the model
    model.evaluate(test_x, test_y)

    # Save the model
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder.joinpath(model_name)
    model.save(model_path)
    print(f"Model saved at {model_path}")
