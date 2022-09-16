import numpy as np
import tensorflow as tf


def _get_tfdataset(features, labels):
    return tf.data.Dataset.from_tensor_slices((features, labels))


def _split_by_class(x, y, number_of_classes):
    """
    Split by class label
    """
    return [x[y == c] for c in range(number_of_classes)]


class AttackData:
    def __init__(
        self,
        m_features, m_labels,
        nm_features, nm_labels,
        num_class,
        attack_percentage,
        batch_size,
        intersection_check=False
    ):

        print(f"Dataset size: {len(m_features)}m-{len(nm_features)}nm")

        if intersection_check:
            assert not any(any(np.array_equal(xm, xnm) for xnm in nm_features)
                           for xm in m_features),\
                "Intersection found between members and nonmembers."

        attack_size = int(attack_percentage * len(m_features))
        assert attack_size > 0,\
            "Attack percentage too low, cannot perform supervised attack."
        assert len(nm_features) >= attack_size,\
            "Attack percentage too high, nonmembers dataset not big enough."
        min_index = min(len(m_features), len(nm_features))

        train_m_features = m_features[:attack_size]
        train_m_labels = m_labels[:attack_size]
        train_nm_features = nm_features[:attack_size]
        train_nm_labels = nm_labels[:attack_size]
        test_m_features = m_features[attack_size:min_index]
        test_m_labels = m_labels[attack_size:min_index]
        test_nm_features = nm_features[attack_size:min_index]
        test_nm_labels = nm_labels[attack_size:min_index]

        mtrain = _get_tfdataset(train_m_features, train_m_labels)
        nmtrain = _get_tfdataset(train_nm_features, train_nm_labels)
        mtest = _get_tfdataset(test_m_features, test_m_labels)
        nmtest = _get_tfdataset(test_nm_features, test_nm_labels)

        print(f"Attack training set size: {len(mtrain)}m-{len(nmtrain)}nm")
        print(f"Attack test     set size: {len(mtest)}m-{len(nmtest)}nm")

        self.mtrain = mtrain.batch(batch_size)
        self.nmtrain = nmtrain.batch(batch_size)
        self.mtest = mtest.batch(batch_size)
        self.nmtest = nmtest.batch(batch_size)

        self.num_class = num_class
        if num_class is not None:
            # Split by class
            test_m_features_byclass = _split_by_class(
                test_m_features, test_m_labels, num_class)
            test_m_labels_byclass = _split_by_class(
                test_m_labels, test_m_labels, num_class)
            test_nm_features_byclass = _split_by_class(
                test_nm_features, test_nm_labels, num_class)
            test_nm_labels_byclass = _split_by_class(
                test_nm_labels, test_nm_labels, num_class)

            test_m_byclass = [_get_tfdataset(x, y) for x, y in zip(
                test_m_features_byclass, test_m_labels_byclass)]
            test_nm_byclass = [_get_tfdataset(x, y) for x, y in zip(
                test_nm_features_byclass, test_nm_labels_byclass)]

            self.mtest_byclass = [z.batch(batch_size) for z in test_m_byclass]
            self.nmtest_byclass = [z.batch(batch_size)
                                   for z in test_nm_byclass]
