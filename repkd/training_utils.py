from typing import Callable

import tensorflow as tf


def generate_MLP(
    architecture: list,
    input_shape: tuple = (784,),
    activation: Callable = tf.nn.sigmoid,
    model_name: str = "no_name_model"
):
    """
    Builds a neural network by the given architecture.
    """

    model = tf.compat.v1.keras.Sequential(
        [
            tf.compat.v1.keras.layers.Dense(
                architecture[0],
                activation=activation,
                input_shape=input_shape,
                kernel_initializer=tf.keras.initializers.LecunNormal(),
                bias_initializer=tf.keras.initializers.RandomNormal(
                    mean=0., stddev=1.)
            ),
        ],
        name=model_name,
    )

    for i in range(1, len(architecture)):
        model.add(
            tf.compat.v1.keras.layers.Dense(
                architecture[i],
                activation=activation,
                kernel_initializer=tf.keras.initializers.LecunNormal(),
                bias_initializer=tf.keras.initializers.RandomNormal(
                    mean=0., stddev=1.)
            )
        )

    return model


class AccuracyThresholdCallback(tf.keras.callbacks.Callback):
    """
    Checks the training accuracy and stop once it reaches the given threshold.
    """

    def __init__(self, accuracy_threshold):
        super(AccuracyThresholdCallback, self).__init__()
        self.accuracy_threshold = accuracy_threshold

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= self.accuracy_threshold:
            self.model.stop_training = True


def train_model(
    model,
    x_train, y_train,
    x_test, y_test,
    accuracy_threshold=None,
    max_epochs=10,
    batch_size=1,
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.SGD(learning_rate=.01),
    verbose="auto"
):

    # Build the model
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # Train the model
    validation_data = (x_test, y_test) if (x_test is not None) else None
    if accuracy_threshold is not None:
        callback = AccuracyThresholdCallback(accuracy_threshold)
        callbacks = [callback]
    else:
        callbacks = []
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=max_epochs,
              validation_data=validation_data,
              callbacks=callbacks,
              shuffle=True,
              verbose=verbose)
