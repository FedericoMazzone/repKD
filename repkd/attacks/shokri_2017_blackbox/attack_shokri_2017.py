import multiprocessing as mp
import multiprocessing.context as ctx
import pickle

import numpy as np
from repkd import utils
from repkd.utils import disable_cuda, log
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from .shadow_model import train_shadow_model


def _split_by_class(x, y, number_of_classes):
    """
    Split by class label
    """
    return [x[y == c] for c in range(number_of_classes)]


def generate_attack_dataset(
    shadow_model_generate,
    shadow_model_train,
    shadow_model_predict,
    full_shadow_train_x, full_shadow_train_y,
    full_shadow_test_x, full_shadow_test_y,
    train_size,
    number_of_classes,
    number_of_shadow_models,
    num_workers=None
):
    """
    shadow_train = full_train but actual_train_sub_set
    shadow_test = full_test but points that will be used for testing the
    attack model
    """

    disable_cuda()

    ctx._force_start_method('spawn')

    num_cpus = mp.cpu_count()
    if not num_workers:
        num_workers = num_cpus - 2
    log(f"Workers: {num_workers}/{num_cpus}")
    p = mp.Pool(num_workers)

    pool_args = [(
        shadow_model_generate,
        shadow_model_train,
        shadow_model_predict,
        full_shadow_train_x, full_shadow_train_y,
        full_shadow_test_x, full_shadow_test_y,
        train_size,
        index
    ) for index in range(number_of_shadow_models)]

    results = p.starmap(train_shadow_model, pool_args)

    full_m_prediction = np.vstack([r[0] for r in results])
    full_nm_prediction = np.vstack([r[1] for r in results])
    full_m_label = np.hstack([r[2] for r in results])
    full_nm_label = np.hstack([r[3] for r in results])

    full_attack_train_x = np.vstack((full_m_prediction, full_nm_prediction))
    full_attack_train_label = np.hstack((full_m_label, full_nm_label))
    full_attack_train_y = np.hstack(
        (
            np.ones(len(full_m_prediction), dtype=np.uint8),
            np.zeros(len(full_nm_prediction), dtype=np.uint8),
        )
    )

    # Split by class label
    attack_train_x = _split_by_class(
        full_attack_train_x, full_attack_train_label, number_of_classes)
    attack_train_y = _split_by_class(
        full_attack_train_y, full_attack_train_label, number_of_classes)

    return attack_train_x, attack_train_y


class AttackModel(list):

    def __init__(self, number_of_classes):
        super().__init__([None]*number_of_classes)
        self.number_of_classes = number_of_classes

    def save(self, outfile):
        """
        Save attack model to output file.

        Parameters
        ----------
        outfile : string
            File to output attack model.
        """
        pickle.dump(self, open(outfile, "wb"))

    @classmethod
    def load(cls, infile):
        """
        Load model from input file.

        Parameters
        ----------
        infile : string
            File from which to load attack model.
        """
        return pickle.load(open(infile, "rb"))

    def fit(self, attack_train_x, attack_train_y):
        pass

    def predict(self, x, y):
        pass

    def evaluate(self, attack_test_x, attack_test_y):

        # Perform predictions
        predictions = [self.predict(attack_test_x[i], i)
                       for i in range(self.number_of_classes)]

        # Evaluation metrics for each class
        for i in range(self.number_of_classes):
            accuracy = accuracy_score(attack_test_y[i], predictions[i])
            precision = precision_score(
                attack_test_y[i], predictions[i], zero_division=0)
            recall = recall_score(
                attack_test_y[i], predictions[i], zero_division=0)
            utils.log(f"Class {i}: accuracy {accuracy:.2f}, precision "
                      f"{precision:.2f}, recall {recall:.2f}")

        # Global evaluation metrics
        predictions = np.hstack(predictions)
        attack_test_y = np.hstack(attack_test_y)
        accuracy = accuracy_score(attack_test_y, predictions)
        precision = precision_score(
            attack_test_y, predictions, zero_division=0)
        recall = recall_score(
            attack_test_y, predictions, zero_division=0)
        utils.log(f"Global : accuracy {accuracy:.2f}, precision "
                  f"{precision:.2f}, recall {recall:.2f}")

        return accuracy


class AttackModelRFC(AttackModel):
    """
    Random Forest Classifier
    """

    def __init__(self, number_of_classes):
        super().__init__(number_of_classes)
        for i in range(number_of_classes):
            self[i] = RandomForestClassifier(n_estimators=100)

    def fit(self, attack_train_x, attack_train_y):
        assert len(attack_train_x) == self.number_of_classes
        assert len(attack_train_y) == self.number_of_classes
        for i in range(self.number_of_classes):
            self[i].fit(attack_train_x[i], attack_train_y[i])

    def predict(self, x, y):
        return self[y].predict(x)


class AttackModelEST(AttackModel):
    """
    Entropy Single Threshold
    """

    def __init__(self, number_of_classes):
        super().__init__(number_of_classes)
        for i in range(number_of_classes):
            self[i] = 0.0

    def fit(self, attack_train_x, attack_train_y):
        assert len(attack_train_x) == self.number_of_classes
        assert len(attack_train_y) == self.number_of_classes
        for i in range(self.number_of_classes):
            entropy_array = entropy(attack_train_x[i], axis=1)
            p = entropy_array.argsort()
            entropy_array = entropy_array[p]
            membership = attack_train_y[i][p]
            total_positives = membership.sum()
            if total_positives == 0:
                optimal_threshold = entropy_array[0] / 2
            elif total_positives == len(membership):
                optimal_threshold = entropy_array[-1] * 2
            else:
                positives = membership.cumsum()
                false_positives = total_positives - positives
                false_negatives = np.arange(len(membership)) + 1 - positives
                falses = false_positives + false_negatives
                # falses = np.abs(false_positives - false_negatives)
                optimal_index = falses.argmin()
                if optimal_index + 1 == len(entropy_array):
                    optimal_threshold = entropy_array[-1] * 2
                else:
                    optimal_threshold = (
                        entropy_array[optimal_index] +
                        entropy_array[optimal_index + 1]
                    ) / 2
            self[i] = optimal_threshold

    def predict(self, x, y):
        prediction = entropy(x, axis=1) < self[y]
        prediction = prediction.astype(np.uint8)
        return prediction


def train_attack_models(
    shadow_model_generate,
    shadow_model_train,
    shadow_model_predict,
    full_shadow_train_x, full_shadow_train_y,
    full_shadow_test_x, full_shadow_test_y,
    train_size,
    number_of_classes,
    number_of_shadow_models=100,
    num_workers=None
):

    # Generate attack models
    attack_model_RFC = AttackModelRFC(number_of_classes)
    attack_model_EST = AttackModelEST(number_of_classes)

    # Generate attack data
    attack_train_x, attack_train_y = generate_attack_dataset(
        shadow_model_generate,
        shadow_model_train,
        shadow_model_predict,
        full_shadow_train_x, full_shadow_train_y,
        full_shadow_test_x, full_shadow_test_y,
        train_size,
        number_of_classes,
        number_of_shadow_models=number_of_shadow_models,
        num_workers=num_workers
    )

    # Train attack models
    attack_model_RFC.fit(attack_train_x, attack_train_y)
    attack_model_EST.fit(attack_train_x, attack_train_y)

    # Return attack models
    return attack_model_RFC, attack_model_EST


def attack(
    target_model,
    target_m_x, target_m_y,
    target_nm_x, target_nm_y,
    number_of_classes,
    attack_model: AttackModel
):
    """
    Attack target model using given attack model.
    """

    m_prediction = target_model.predict(target_m_x)
    nm_prediction = target_model.predict(target_nm_x)
    full_attack_test_x = np.vstack((m_prediction, nm_prediction))
    full_attack_test_label = np.hstack((target_m_y, target_nm_y))
    full_attack_test_y = np.hstack(
        (
            np.ones(len(m_prediction), dtype=np.uint8),
            np.zeros(len(nm_prediction), dtype=np.uint8),
        )
    )
    attack_test_x = _split_by_class(
        full_attack_test_x, full_attack_test_label, number_of_classes)
    attack_test_y = _split_by_class(
        full_attack_test_y, full_attack_test_label, number_of_classes)

    return attack_model.evaluate(attack_test_x, attack_test_y)
