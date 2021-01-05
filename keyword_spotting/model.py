import tensorflow as tf
from kapre.composed import get_melspectrogram_layer
from tcn import TCN
from tensorflow.keras import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import (Activation, AveragePooling2D,
                                            BatchNormalization, Conv2D, Dense,
                                            Dropout, Flatten, Input, Lambda,
                                            SpatialDropout2D)
from tensorflow.python.ops import math_ops

from keyword_spotting.data import SAMPLE_RATE


class Sum1(tf.keras.constraints.Constraint):
    """Constrains the weights incident to each hidden unit to have unit norm.
    Arguments:
      axis: integer, axis along which to calculate weight norms.
          For instance, in a `Dense` layer the weight matrix
          has shape `(input_dim, output_dim)`,
          set `axis` to `0` to constrain each weight vector
          of length `(input_dim,)`.
          In a `Conv2D` layer with `data_format="channels_last"`,
          the weight tensor has shape
          `(rows, cols, input_depth, output_depth)`,
          set `axis` to `[0, 1, 2]`
          to constrain the weights of each filter tensor of size
          `(rows, cols, input_depth)`.
     """

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, w):
        return w / (
            K.epsilon() +
            math_ops.reduce_sum(w, axis=self.axis, keepdims=True))

    def get_config(self):
        return {'axis': self.axis}


class DNNModel:
    def __init__(self, input_shape, number_of_classes):
        self.build_model(input_shape, number_of_classes)

    def build_model(self, input_shape, number_of_classes):
        pass

    def description(self):
        pass


def get_model(input_shape, number_of_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(number_of_classes, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')
    return model


def ExpandDimension():
    return Lambda(lambda x: K.expand_dims(x))


def get_model_2(input_shape, number_of_classes):
    input = Input(shape=input_shape)
    x = input
    x = get_melspectrogram_layer(n_fft=1024,
                                 pad_begin=True,
                                 hop_length=128, input_shape=(SAMPLE_RATE, 1),
                                 sample_rate=SAMPLE_RATE, n_mels=80,
                                 mel_f_min=40.0, mel_f_max=SAMPLE_RATE/2,
                                 return_decibel=True,
                                 name='mel_stft')(x)

    x = Conv2D(16,
               strides=(1, 3),
               kernel_size=(3, 7),
               padding='same',
               use_bias=False)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.3)(x)

    x = Flatten()(x)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(number_of_classes, activation='softmax',
              kernel_regularizer=tf.keras.regularizers.L2(0.001))(x)
    model = Model(inputs=[input], outputs=[x])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')
    return model


def get_model_tcn(input_shape, number_of_classes):
    input = Input(shape=input_shape)
    x = input
    x = TCN(32, use_skip_connections=True,
            dropout_rate=0.3, use_batch_norm=True)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(number_of_classes, activation='softmax')(x)
    model = Model(inputs=[input], outputs=[x])
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001,  clipnorm=0.001, clipvalue=0.001)
    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')
    return model
