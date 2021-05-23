import argparse

from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf
import yaml
from keyword_spotting.feature_extraction.utils import (
    extract_features as keyword_extract_features,
)
from keyword_spotting.feature_extraction.utils import read_wav
from keyword_spotting.model import cnn_inception2, models

noise_files = [
    "doing_the_dishes.wav",
    "dude_miaowing.wav",
    "exercise_bike.wav",
    "pink_noise.wav",
    "running_tap.wav",
    "white_noise.wav",
]


labels = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "unknown",
    "silence",    
]

def create_data_path(dataset_path: Path, data):
    return [f'{dataset_path}/{x}' for x in data]

def load_data(dataset_path: Path):

    with open(dataset_path / f'X_train.pickle', 'rb') as f:
        X_train = pickle.load(f)
    X_train = create_data_path(dataset_path, X_train)

    with open(dataset_path / f'X_val.pickle', 'rb') as f:
        X_val = pickle.load(f)
    X_val = create_data_path(dataset_path, X_val)

    with open(dataset_path / f'X_test.pickle', 'rb') as f:
        X_test = pickle.load(f)
    X_test = create_data_path(dataset_path, X_test)
    
    return X_train, X_val, X_test

lables_dict = {l: i for i, l in enumerate(labels)}


def tf_extract_features(sample_rate, signal, label):
    def process_file(sample_rate, signal, label):
        return keyword_extract_features(sample_rate, signal), label

    return tf.numpy_function(
        process_file, [sample_rate, signal, label], [tf.float32, tf.int32]
    )


def tf_read_wav(path):
    def read_wav_(file_path):
        file_path = bytes.decode(file_path)
        label = file_path.split("/")[-2]
        fs, data = read_wav(file_path)
        return fs, np.float32(data), np.int32(lables_dict[label])

    return tf.numpy_function(read_wav_, [path], [tf.int64, tf.float32, tf.int32])


def tf_windowed(data, label, left: int = 23, right: int = 8):
    def windowed_(data, label):
        d = np.array(
            [data[j - left : j + right, :] for j in range(left, data.shape[0] - right)],
            dtype=np.float32,
        )
        labels = np.array(
            [label for j in range(left, data.shape[0] - right)], dtype=np.int32
        )
        return d, labels

    return tf.numpy_function(windowed_, [data, label], [tf.float32, tf.int32])


def tf_add_noise(sample_rate, signal, label):
    def add_noise(sample_rate, signal, label):
        if np.random.rand() > 0.8:
            noise_path = (
                data_path / "_background_noise_" / np.random.choice(noise_files)
            )
            fs, data_noise = read_wav(noise_path)
            signal = signal + data_noise[: len(signal)]
        return sample_rate, signal, label

    return tf.numpy_function(
        add_noise, [sample_rate, signal, label], [tf.int64, tf.float32, tf.int32]
    )


def split_window(x, y):

    return tf.data.Dataset.from_tensor_slices((x, y))


def shapeify(x, y):

    x.set_shape([31, 12])
    y.set_shape([])

    return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--config", type=str, required=True, help="Config file")

    args = parser.parse_args()

    config = None
    with open(args.config, "r") as file:
        config = yaml.load(file.read(), Loader=yaml.SafeLoader)

    data_path = Path(config["dataset"]["path"])

    X_train, X_val, X_test = load_data(data_path)
    print(X_train[0])
    
    ds_train = (
        
        tf.data.Dataset.from_tensor_slices(X_train)
        .shuffle(buffer_size=5000)
        .map(tf_read_wav)
        .apply(tf.data.experimental.ignore_errors())
        .map(tf_add_noise)
        .map(tf_extract_features)
        .map(tf_windowed)
        .flat_map(split_window)
        .map(shapeify)
        .prefetch(tf.data.AUTOTUNE)
    )
    number_of_classes = len(labels)
    input_shape = [a[0].shape for a in ds_train.take(1)][0]
    params = config["model"].get("params", {})
    model = models[config["model"]["name"]](input_shape, number_of_classes, **params)
    model.summary()
    epochs = config["train"]["epochs"]

    model_path = Path(config["model"]["path"]).resolve()

    # early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)
    # check_point = tf.keras.callbacks.ModelCheckpoint(model_filename)
    history = model.fit(ds_train.batch(512), epochs=epochs)

    # model = cnn_inception2((input_shape[0], input_shape[1]), 10)
    # model.summary()
    # model.fit(ds, epochs=5)
