from pathlib import Path

import numpy as np


def save_data_subset(
    dataset,
    subset_size,
    data_dir: Path,
    data_type,
    uniform=False
):

    assert data_type in ["train", "test", "validation"]

    uniform_log = ""
    uniform_name = ""

    if uniform:
        classes = np.unique(dataset[:, -1])
        num_class = len(classes)
        assert subset_size % num_class == 0, \
            "Subset size is not a multiple of the number of classes."
        num_samples_per_class = subset_size // num_class
        max_index = 0
        subset = np.zeros(
            (subset_size, dataset.shape[-1]), dtype=dataset.dtype)
        # interleaving
        for c_index, c in enumerate(classes):
            tmp = dataset[dataset[:, -1] == c][:num_samples_per_class]
            subset[c_index::num_class] = tmp
            max_index = max(max_index,
                            np.where((dataset == tmp[-1]).all(1))[0][0])
        uniform_log = f" Max index used to guarantee uniformity: {max_index}"
        uniform_name = "_uniform"
    else:
        subset = dataset[:subset_size]

    file_name = f"{data_type}{uniform_name}_subset_{subset_size}"
    file_path = data_dir.joinpath(file_name)
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(file_path, subset)

    print(f"Saved {data_type} set at {file_path}.{uniform_log}")
