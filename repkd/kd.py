import tensorflow as tf

from repkd import utils


class SaveBestModel(tf.keras.callbacks.Callback):

    def __init__(self, val_x, val_y):
        self.val_x = val_x
        self.val_y = val_y
        self.best_acc = 0.0

    def on_train_batch_end(self, batch, logs=None):
        _, val_acc = self.model.evaluate(self.val_x, self.val_y, verbose=0)
        utils.log(f"{batch=} - {val_acc=}")
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            utils.log(f"New best model found! {val_acc=}")
            self.best_weights = self.model.get_weights()


def kd(
    teacher_model,
    original_train_x, original_train_y,
    original_test_x, original_test_y,
    surrogate_data_x,
    generate_model, generate_autoencoder=None,
    original_val_x=None, original_val_y=None,
    conf_masking=None,
    pt_loss="mean_squared_error",
    pt_optimizer=tf.keras.optimizers.Adam(),
    pt_batch=100,
    pt_epochs=100,
    ft_loss="categorical_crossentropy",
    ft_optimizer=tf.keras.optimizers.Adam(),
    ft_batch=10,
    ft_epochs=10,
    bm_batch=100,
    bm_epochs=1
):

    utils.log("Start knowledge distillation block")

    utils.log(f"Surrogate data size: {len(surrogate_data_x)}")

    # Evaluating teacher model
    _, teacher_train_acc = teacher_model.evaluate(
        original_train_x, original_train_y, verbose=0)
    _, teacher_test_acc = teacher_model.evaluate(
        original_test_x, original_test_y, verbose=0)
    utils.log("Teacher model accuracy: "
              f"train={teacher_train_acc}, test={teacher_test_acc}")

    # Generate student model
    utils.log("Generate student model")
    student_model = generate_model()

    # Pre-training
    if generate_autoencoder is None:
        utils.log("No pre-training")
    else:
        utils.log("Pre-training")
        autoencoder = generate_autoencoder()
        autoencoder.compile(
            loss=pt_loss,
            optimizer=pt_optimizer
        )
        autoencoder.fit(
            surrogate_data_x, surrogate_data_x,
            batch_size=pt_batch,
            epochs=pt_epochs,
            shuffle=True
        )
        # Copy encoder into student model
        for i in range(len(student_model.layers)):
            student_model.layers[i].set_weights(
                autoencoder.layers[i].get_weights())

    # Label surrogate dataset and apply confidence masking
    utils.log("Labeling surrogate dataset")
    surrogate_data_y = teacher_model.predict(surrogate_data_x)
    if conf_masking is None:
        utils.log("No masking function provided")
    else:
        utils.log("Masking predictions")
        surrogate_data_y = conf_masking(surrogate_data_y)

    # Fine-tuning
    utils.log("Fine-tuning")
    student_model.compile(
        loss=ft_loss,
        optimizer=ft_optimizer,
        metrics=['accuracy']
    )
    student_model.fit(
        surrogate_data_x, surrogate_data_y,
        batch_size=ft_batch,
        epochs=ft_epochs,
        validation_data=(original_test_x, original_test_y),
        shuffle=True
    )

    # Create save best model callback
    if (original_val_x is None) or (original_val_y is None):
        utils.log("No validation set")
    else:
        utils.log("Validation set provided")
        save_best_model = SaveBestModel(original_val_x, original_val_y)
        # Search for best model
        student_model.fit(
            surrogate_data_x, surrogate_data_y,
            batch_size=bm_batch,
            epochs=bm_epochs,
            callbacks=[save_best_model],
            verbose=0
        )
        # Set best weights
        student_model.set_weights(save_best_model.best_weights)

    # Test student model
    _, student_train_acc = student_model.evaluate(
        original_train_x, original_train_y, verbose=0)
    _, student_test_acc = student_model.evaluate(
        original_test_x, original_test_y, verbose=0)
    utils.log("Student model accuracy: "
              f"train={student_train_acc}, test={student_test_acc}")

    return student_model
