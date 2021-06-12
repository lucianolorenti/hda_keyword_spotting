import argparse
from time import time
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
from tqdm.auto import tqdm
import logging
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from python_speech_features import mfcc
import librosa

logging.basicConfig()
logger = logging.getLogger("hda")

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


lables_dict = {l: i for i, l in enumerate(labels)}

def dct(n_filters, n_input):
    basis = np.empty((n_filters, n_input))
    basis[0, :] = 1.0 / np.sqrt(n_input)

    samples = np.arange(1, 2*n_input, 2) * np.pi / (2.0 * n_input)

    for i in range(1, n_filters):
        basis[i, :] = np.cos(i*samples) * np.sqrt(2.0/n_input)

    return basis

dct_filters = dct(40, 40)

def tf_extract_features(sample_rate, signal, label):
    def process_file(sample_rate, signal, label):
        data = librosa.feature.melspectrogram(
            signal.astype(np.float32),
            sr=sample_rate,
            n_mels=40,
            hop_length=sample_rate // 1000 * 10,
            n_fft=480,
            fmin=20,
            fmax=4000,
        )
        data[data > 0] = np.log(data[data > 0])
        data = [np.matmul(dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]

        data = np.array(data, order="F").astype(np.float32)      
        return data.reshape(-1, 40), label
        # return mfcc(signal, sample_rate, numcep=12, nfft=512).astype('float32'), label

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


def windowed_(data, label, left: int = 30, right: int = 10):
    d = np.array(
        [data[j - left : j + right, :] for j in range(left, data.shape[0] - right)],
        dtype=np.float32,
    )
    labels = np.array(
        [label for j in range(left, data.shape[0] - right)], dtype=np.int32
    )
  
    return d, labels


def tf_windowed(data, label, left: int = 30, right: int = 10):
    return tf.numpy_function(
        lambda data, label: windowed_(data, label, left, right),
        [data, label],
        [tf.float32, tf.int32],
    )


def tf_add_noise(sample_rate, signal, label):
    def add_noise(sample_rate, signal, label):
        if np.random.rand() > 0.8:
            noise_path = (
                data_path / "_background_noise_" / np.random.choice(noise_files)
            )
            fs, data_noise = read_wav(noise_path)
            min_length = min(data_noise.shape[0], signal.shape[0])
            noise_factor = np.random.rand() * 0.1
            signal[:min_length] = (
                signal[:min_length] + noise_factor * data_noise[:min_length]
            )
        return sample_rate, signal, label

    return tf.numpy_function(
        add_noise, [sample_rate, signal, label], [tf.int64, tf.float32, tf.int32]
    )


def split_window(x, y):

    return tf.data.Dataset.from_tensor_slices((x, y))


def shapeify(x, y):

    x.set_shape([40, 40])
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
    output_path = Path(config["output_path"])

    X_train, X_val, X_test = load_data(data_path)

    ds_train = (
        tf.data.Dataset.from_tensor_slices(X_train[:1000])
        .map(tf_read_wav, num_parallel_calls=4)
        .apply(tf.data.experimental.ignore_errors())
        .map(tf_add_noise)
        .map(tf_extract_features, num_parallel_calls=4)
        .map(tf_windowed)
        .flat_map(split_window)
        .shuffle(buffer_size=8000)
        .map(shapeify)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    # asd=np.sum(1 for _ in ds_train)
    # print(asd)

    number_of_classes = len(labels)
    # input_shape = [a[0].shape for a in ds_train.take(1)][0]
    input_shape = [40, 40]
    params = config["model"].get("params", {})

    model = models[config["model"]["name"]](input_shape, number_of_classes, **params)
    model.summary()
    epochs = config["train"]["epochs"]

    model_path = Path(config["model"]["path"]).resolve()

    batch_size = config["train"]["batch_size"]

    ds_val = (
        tf.data.Dataset.from_tensor_slices(X_val[:15])
        .map(tf_read_wav)
        .apply(tf.data.experimental.ignore_errors())
        .prefetch(tf.data.AUTOTUNE)
        .map(tf_extract_features)
        .map(tf_windowed)
        .flat_map(split_window)
        .map(shapeify)
        .prefetch(tf.data.AUTOTUNE)
    )

    start = time()
    history = model.fit(
        ds_train.batch(batch_size),
        validation_data=ds_val.batch(batch_size),
        epochs=epochs,
        # steps_per_epoch=asd // batch_size,
        callbacks=[EarlyStopping(patience=5), ReduceLROnPlateau(patience=1, verbose=1)],
    )
    total_time = time() - start

    results = []
    for audio_file in tqdm(X_test):
        try:
            label = Path(audio_file).resolve().parts[-2]
            label = lables_dict[label]
            sample_rate, signal = read_wav(audio_file)
            data = keyword_extract_features(sample_rate, signal)
            data, labels = windowed_(data, label)
            predicted = model.predict(data)
            results.append((audio_file, label, predicted))
        except:
            pass

    with open(output_path, "wb") as file:
        pickle.dump((config, total_time, results), file)

    model.save_weights(str(model_path))
