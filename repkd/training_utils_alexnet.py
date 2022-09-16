# Code taken from
# https://github.com/privacytrustlab/ml_privacy_meter/blob/master/tutorials/alexnet.py

import tensorflow as tf


def generate_AlexNet(
    input_shape: tuple = (32, 32, 3),
    model_name: str = "no_name_model"
):
    """
    AlexNet:
    Described in: http://arxiv.org/pdf/1404.5997v2.pdf
    Parameters from:
    github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
    """
    # Creating initializer, optimizer and the regularizer ops
    initializer = tf.compat.v1.keras.initializers.random_normal(0.0, 0.01)
    regularizer = tf.keras.regularizers.l2(5e-4)

    inputshape = (input_shape[0], input_shape[1], input_shape[2],)

    # Creating the model
    model = tf.compat.v1.keras.Sequential(
        [
            tf.compat.v1.keras.layers.Conv2D(
                64, 11, 4,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                input_shape=inputshape,
                data_format='channels_last'
            ),
            tf.compat.v1.keras.layers.MaxPooling2D(
                2, 2, padding='valid'
            ),
            tf.compat.v1.keras.layers.Conv2D(
                192, 5,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.nn.relu
            ),
            tf.compat.v1.keras.layers.MaxPooling2D(
                2, 2, padding='valid'
            ),
            tf.compat.v1.keras.layers.Conv2D(
                384, 3,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.nn.relu
            ),
            tf.compat.v1.keras.layers.Conv2D(
                256, 3,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.nn.relu
            ),
            tf.compat.v1.keras.layers.Conv2D(
                256, 3,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.nn.relu
            ),
            tf.compat.v1.keras.layers.MaxPooling2D(
                2, 2, padding='valid'
            ),
            tf.compat.v1.keras.layers.Flatten(),
            tf.compat.v1.keras.layers.Dropout(0.3),
            tf.compat.v1.keras.layers.Dense(
                100,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                activation=tf.nn.softmax
            )
        ]
    )
    return model


# def scheduler(epoch):
#     """
#     Learning rate scheduler
#     """
#     lr = 0.0001
#     if epoch > 25:
#         lr = 0.00001
#     elif epoch > 60:
#         lr = 0.000001
#     print('Using learning rate', lr)
#     return lr

def scheduler(epoch):
    """
    Learning rate scheduler
    """
    lr = 0.001
    if epoch > 100:
        lr = 0.0001
    elif epoch > 125:
        lr = 0.00001
    print('Using learning rate', lr)
    return lr


def normalize(f, means, stddevs):
    """
    Normalizes data using means and stddevs
    """
    normalized = (f/255 - means) / stddevs
    return normalized
