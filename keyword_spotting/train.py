from keyword_spotting.utils import add_noise
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf


from keyword_spotting.feature_extraction.utils import (
    extract_features as keyword_extract_features,
)
from keyword_spotting.feature_extraction.utils import read_wav, windowed
from keyword_spotting.predictions import (
    labels_dict,
)






def create_data_path(dataset_path: Path, data):
    return [f"{dataset_path}/{x}" for x in data]


def load_data(dataset_path: Path):

    with open(dataset_path / f"X_train.pickle", "rb") as f:
        X_train = pickle.load(f)
    X_train = create_data_path(dataset_path, X_train)

    with open(dataset_path / f"X_val.pickle", "rb") as f:
        X_val = pickle.load(f)
    X_val = create_data_path(dataset_path, X_val)

    with open(dataset_path / f"X_test.pickle", "rb") as f:
        X_test = pickle.load(f)
    X_test = create_data_path(dataset_path, X_test)

    return X_train, X_val, X_test


def tf_extract_features(sample_rate, signal, label):
    def extract_features_(sample_rate, signal, label):
        a = keyword_extract_features(signal, sample_rate)
        b = np.zeros((100, 40), dtype=np.float32)
        b[: a.shape[0], :] = a
        return b, label

    return tf.numpy_function(
        extract_features_, [sample_rate, signal, label], [tf.float32, tf.int32]
    )


def tf_read_wav(path):
    def read_wav_(file_path):
        file_path = bytes.decode(file_path)
        label = file_path.split("/")[-2]
        fs, data = read_wav(file_path)
        return fs, np.float32(data), np.int32(labels_dict[label])

    return tf.numpy_function(read_wav_, [path], [tf.int64, tf.float32, tf.int32])


def tf_windowed(data, label, left: int = 30, right: int = 10):
    return tf.numpy_function(
        lambda data, label: windowed(data, label, left, right),
        [data, label],
        [tf.float32, tf.int32],
    )


def tf_add_noise(sample_rate, signal, label, dataset_path:Path):
    def add_noise_(sample_rate, signal, label):        
        _, signal =  add_noise(sample_rate, signal, dataset_path)
        return sample_rate, signal, label

    return tf.numpy_function(
        add_noise_, [sample_rate, signal, label], [tf.int64, tf.float32, tf.int32]
    )


def split_window(x, y):
    return tf.data.Dataset.from_tensor_slices((x, y))


def shapify_not_windowed(x, y):
    x.set_shape([100, 40])
    y.set_shape([])
    return x, y


def shapify_windowed(x, y):
    x.set_shape([40, 40])
    y.set_shape([])
    return x, y


def build_dataset_generator(
    dataset,
    dataset_path: Path,
    windowed: bool = False,
    noise: bool = False,
    shuffle: bool = True,
):
    gen_dataset = (
        tf.data.Dataset.from_tensor_slices(dataset)
        .map(tf_read_wav, num_parallel_calls=4)
        .apply(tf.data.experimental.ignore_errors())
    )
    if noise:
        gen_dataset = gen_dataset.map(
            lambda sample_rate, signal, label: tf_add_noise(
                sample_rate, signal, label, dataset_path
            )
        )

    gen_dataset = gen_dataset.map(tf_extract_features, num_parallel_calls=4)

    if windowed:
        shapify_func = shapify_windowed
        gen_dataset = gen_dataset.map(tf_windowed).flat_map(split_window)
    else:
        shapify_func = shapify_not_windowed
    if shuffle:
        gen_dataset = gen_dataset.shuffle(buffer_size=500)

    gen_dataset = gen_dataset.map(shapify_func).prefetch(tf.data.AUTOTUNE)
    return gen_dataset
