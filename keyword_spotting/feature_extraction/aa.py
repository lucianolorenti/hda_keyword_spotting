import tensorflow as tf
from pathlib import Path
from keyword_spotting.feature_extraction.extractor import extract_features as keyword_extract_features
from keyword_spotting.feature_extraction.utils import read_wav
from keyword_spotting.model import cnn_inception2

import numpy as np
import tensorflow as tf
from kapre.composed import get_melspectrogram_layer
from tcn import TCN
from tensorflow.keras import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import (GRU, Activation, Add,
                                            AveragePooling2D,
                                            BatchNormalization, Bidirectional,
                                            Concatenate, Conv2D, Dense, Dot,
                                            Dropout, Flatten, Input, Lambda,
                                            LayerNormalization, Reshape,
                                            Softmax, SpatialDropout2D)
from tensorflow.python.ops import math_ops

from keyword_spotting.data import SAMPLE_RATE



noise_files = [
    'doing_the_dishes.wav', 'dude_miaowing.wav', 'exercise_bike.wav',
    'pink_noise.wav', 'running_tap.wav', 'white_noise.wav'
]

labels = ['down', 'go' ,'left','no','off','on','right','stop','up','yes']
lables_dict = {l:i for i,l in enumerate(labels)}


def tf_extract_features(sample_rate, signal, label):
    def process_file(sample_rate, signal, label):
        return keyword_extract_features(sample_rate, signal), label

    return tf.numpy_function(process_file, [sample_rate, signal, label], [tf.float32, tf.int32])


def tf_read_wav(path):
    def read_wav_(file_path):
        file_path = bytes.decode(file_path)
        label = file_path.split('/')[-2]
        fs, data = read_wav(file_path)
        return fs, np.float32(data), np.int32(lables_dict[label])

    return tf.numpy_function(read_wav_, [path], [tf.int64, tf.float32, tf.int32])


def tf_windowed(data, label, left: int = 23, right: int = 8):
    def windowed_(data, label):
        d =  np.array([
            data[j - left:j + right, :]
            for j in range(left, data.shape[0] - right)
        ],
                        dtype=np.float32)
        labels = np.array([label for j in range(left, data.shape[0] - right)], dtype=np.int32)
        return d, labels

    return tf.numpy_function(windowed_, [data, label], [tf.float32 , tf.int32])


def tf_add_noise(sample_rate, signal, label):
    def add_noise(sample_rate, signal, label):
        if np.random.rand() > 0.8:
            noise_path = data_path / '_background_noise_' /  np.random.choice(noise_files)
            fs, data_noise = read_wav(noise_path)
            signal = signal + data_noise[:len(signal)]
        return sample_rate, signal, label

    return tf.numpy_function(add_noise, [sample_rate, signal, label],
                             [tf.int64, tf.float32, tf.int32])


def split_window(x, y):
    
    return tf.data.Dataset.from_tensor_slices((x,y))


def shapeify(x, y):

    x.set_shape([31, 12])
    y.set_shape([])

    return x, y

data_path = Path('/home/luciano/speech')

ds = (tf.data.Dataset.list_files('/home/luciano/speech/dd/*/*.wav')
        #.filter(tf_filter_files)
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

input_shape = [a[0].shape for a in ds.take(1)][0]

#model = cnn_inception2((input_shape[0], input_shape[1]), 10)
#model.summary()
#model.fit(ds, epochs=5)
input = Input(shape=(input_shape[0], input_shape[1]))
x = input
x = Flatten()(x)
x = Dense(50, activation='relu')(x)
x = Dense(10, activation='softmax')(x)
model = Model(inputs=[input], outputs=[x])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
                metrics=['accuracy'],
                loss='sparse_categorical_crossentropy')
model.fit(ds.batch(64), batch_size=64)