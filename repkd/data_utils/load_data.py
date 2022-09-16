import numpy as np
from tensorflow.keras.utils import to_categorical


def load_data(infile, number_of_labels=None, normalize=True, shape=None):
    """
    Load and prepare data. The function assumes the data to be stored as a
    numpy matrix, each row containing a datapoint features concatenated to the
    label value.

    Parameters
    ----------
    infile : string
        File from which to load data.
    number_of_labels : int
        Number of expected labels.
    normalize : bool
        If true, normalize bytes to floats in [0, 1).
    shape : tuple
        Reshape features to match the given shape.

    Returns
    -------
    features : numpy array
        Feature array of data stored in the file.
    labels : numpy array
        Label array of data stored in the file.

    Examples
    --------
    Load MNIST dataset from saved location, with tensor labels, and byte-value
    features shaped as 28x28.

    >>> train_x, train_y = load_dataset(
    ...     "data/mnist_digits/train_subset_400.npy", 10,
    ...     normalize=False, shape=(-1, 28, 28))

    """
    dataset = np.load(infile)
    features, labels = dataset[:, :-1], dataset[:, -1]
    if number_of_labels:
        labels = to_categorical(labels, number_of_labels)
    if normalize:
        features = features.astype(np.float32) / 255
    if shape:
        features = features.reshape(shape)
    return features, labels
