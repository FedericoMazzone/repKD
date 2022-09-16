from repkd.utils import log


def _data_choice(data_x, data_y, number_of_samples):
    from numpy.random import choice
    assert len(data_x) == len(data_y)
    indexes = choice(
        range(len(data_x)), number_of_samples, replace=False)
    return data_x[indexes], data_y[indexes]


def train_shadow_model(
        model_generate,
        model_train,
        model_predict,
        full_shadow_train_x, full_shadow_train_y,
        full_shadow_test_x, full_shadow_test_y,
        train_size,
        index=-1
):
    """
    Generate and train a shadow model, then use it to predict on members and
    non-members.

    Parameters
    ----------
    model_generate : function
        Function that generates a shadow model.
    model_train : function
        Function that trains a shadow model.
    model_predict : function
        Function that predicts out of a shadow model.
    full_shadow_train_x : array
    full_shadow_train_y : array
    full_shadow_test_x : array
    full_shadow_test_y : array
    train_size : int
    Returns
    -------

    """

    log(f"Shadow model {index}: start")

    # Generate shadow model
    shadow_model = model_generate()

    # Choose train and test data for shadow model
    shadow_train_x, shadow_train_y = _data_choice(
        full_shadow_train_x, full_shadow_train_y, train_size)
    shadow_test_x, shadow_test_y = _data_choice(
        full_shadow_test_x, full_shadow_test_y, train_size)

    # Train shadow model
    model_train(shadow_model, shadow_train_x, shadow_train_y)

    # Predict
    train_prediction = model_predict(shadow_model, shadow_train_x)
    test_prediction = model_predict(shadow_model, shadow_test_x)

    log(f"Shadow model {index}: end")

    # Return predictions along with class labels
    return train_prediction, test_prediction, shadow_train_y, shadow_test_y
