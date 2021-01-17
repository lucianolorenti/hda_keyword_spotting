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


def ExpandDimension():
    return Lambda(lambda x: K.expand_dims(x))


def cnn_residual2(input_shape,  number_of_classes, n_residuals=3):
    """
    Convolutional Neural Networks for Small-footprint Keyword Spotting
    Tara N. Sainath, Carolina Parada
    """
    input = Input(shape=input_shape)
    x = input
    x = LayerNormalization(axis=1)(x)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = ExpandDimension()(x)
    identity = x
    identity = Conv2D(45,
                      strides=(1, 1),
                      kernel_size=(3, 3),
                      padding='same',
                      use_bias=False)(x)
    i = 0
    for j in range(n_residuals):
        i += 1
        x = Conv2D(45,
                   strides=(1, 1),
                   kernel_size=(3, 3),
                   dilation_rate=int(2**(i // 3)),
                   padding='same',
                   use_bias=False)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        i = 2
        x = Conv2D(45,
                   strides=(1, 1),
                   kernel_size=(3, 3),
                   dilation_rate=int(2**(i // 3)),
                   padding='same',
                   use_bias=False)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        x = Add()([x, identity])

    x = Conv2D(64,
               strides=(1, 1),
               kernel_size=(3, 3),
               padding='same',
               use_bias=True)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D()(x)

    output = Dense(number_of_classes, activation='softmax', name='output')(x)

    model = Model(inputs=[input], outputs=[output])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')
    return model


def cnn_residual(input_shape, number_of_classes):
    """
    Convolutional Neural Networks for Small-footprint Keyword Spotting
    Tara N. Sainath, Carolina Parada
    """
    input = Input(shape=input_shape)
    x = input
    x = LayerNormalization(axis=1)(x)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = ExpandDimension()(x)
    identity = x
    identity = Conv2D(64,
                      strides=(1, 3),
                      kernel_size=(1, 1),
                      padding='same',
                      use_bias=False)(x)

    x = Conv2D(64,
               strides=(1, 3),
               kernel_size=(20, 8),
               padding='same',
               use_bias=False)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64,
               strides=(1, 1),
               kernel_size=(10, 4),
               padding='same',
               use_bias=False)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Add()([x, identity])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64,
               strides=(1, 1),
               kernel_size=(1, 1),
               padding='same',
               use_bias=True)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)

    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    output = Dense(number_of_classes, activation='softmax', name='output')(x)

    model = Model(inputs=[input], outputs=[output])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')
    return model


def cnn_attention(input_shape, number_of_classes):
    """
    Convolutional Neural Networks for Small-footprint Keyword Spotting
    Tara N. Sainath, Carolina Parada
    """
    input = Input(shape=input_shape)
    x = input
    x = LayerNormalization(axis=1)(x)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = ExpandDimension()(x)

    x = Conv2D(64,
               strides=(1, 3),
               kernel_size=(20, 8),
               padding='same',
               use_bias=False)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64,
               strides=(1, 1),
               kernel_size=(10, 4),
               padding='same',
               use_bias=False)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Reshape((x.shape[1], x.shape[2]*x.shape[3]))(x)
    new_shape = x.shape

    xFirst = Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = Dense(new_shape[2])(xFirst)

    # dot product attention
    attScores = Dot(axes=[1, 2])([query, x])
    attScores = Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = Dense(64)(attVector)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    output = Dense(number_of_classes, activation='softmax', name='output')(x)

    model = Model(inputs=[input], outputs=[output])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')
    return model


def simple_tcn(input_shape, number_of_classes):
    """
    Convolutional Neural Networks for Small-footprint Keyword Spotting
    Tara N. Sainath, Carolina Parada
    """
    input = Input(shape=input_shape)
    x = input

    x = TCN(64, dilations=[1, 2, 4, 8, 16, 32],
            return_sequences=True, use_batch_norm=True, use_skip_connections=True)(x)
    x = TCN(64, dilations=[1, 2, 4, 8, 16, 32],
            return_sequences=True, use_batch_norm=True, use_skip_connections=True)(x)
    x = Flatten()(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    output = Dense(number_of_classes, activation='softmax', name='output')(x)

    model = Model(inputs=[input], outputs=[output])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')
    return model


def cnn_inception(input_shape, number_of_classes):
    """
    Convolutional Neural Networks for Small-footprint Keyword Spotting
    Tara N. Sainath, Carolina Parada
    """
    input = Input(shape=input_shape)
    x = input
    x = ExpandDimension()(x)

    x1 = Conv2D(16,
                strides=(1, 3),
                kernel_size=(10, 8),
                padding='same',
                use_bias=False)(x)

    x2 = Conv2D(16,
                strides=(1, 3),
                kernel_size=(20, 8),
                padding='same',
                use_bias=False)(x)

    x3 = Conv2D(16,
                strides=(1, 3),
                kernel_size=(30, 8),
                padding='same',
                use_bias=False)(x)
    x = Concatenate()([x1, x2, x3])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64,
               strides=(1, 1),
               kernel_size=(10, 4),
               padding='same',
               use_bias=False)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.3)(x)

    x = Flatten()(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(number_of_classes, activation='softmax')(x)
    model = Model(inputs=[input], outputs=[x])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')
    return model


def cnn_trad_fpool3(input_shape, number_of_classes):
    """
    Convolutional Neural Networks for Small-footprint Keyword Spotting
    Tara N. Sainath, Carolina Parada
    """
    input = Input(shape=input_shape)
    x = input
    x = ExpandDimension()(x)

    x = Conv2D(64,
               strides=(1, 3),
               kernel_size=(20, 8),
               padding='same',
               use_bias=False)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64,
               strides=(1, 1),
               kernel_size=(10, 4),
               padding='same',
               use_bias=False)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(number_of_classes, activation='softmax')(x)
    model = Model(inputs=[input], outputs=[x])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')
    return model


models = {
    'cnn_trad_fpool3': cnn_trad_fpool3,
    'simple_tcn': simple_tcn,
    'cnn_attention': cnn_attention,
    'cnn_residual': cnn_residual,
    'cnn_inception': cnn_inception,
    'cnn_residual2': cnn_residual2
}
