import tensorflow as tf
from kapre.composed import get_melspectrogram_layer
from tcn import TCN
from tensorflow.keras import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import (Activation, AveragePooling2D,
                                            BatchNormalization, Bidirectional,
                                            Conv2D, Dense, Dot, Dropout,
                                            Flatten, Input, Lambda,
                                            LayerNormalization, ReLU, Softmax,
                                            SpatialDropout2D)
from tensorflow.python.ops import math_ops

from keyword_spotting.data import SAMPLE_RATE


def ExpandDimension():
    return Lambda(lambda x: K.expand_dims(x))


def cnn_rnn_attention(input_shape, number_of_classes):
    """
    Convolutional Neural Networks for Small-footprint Keyword Spotting
    Tara N. Sainath, Carolina Parada
    """
    input = Input(shape=input_shape)
    x = input
    x = LayerNormalization(axis=0)(x)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = ExpandDimension()(x)

    x = Conv2D(10, (5, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # x = Reshape((125, 80)) (x)
    # keras.backend.squeeze(x, axis)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(x)

    x = Bidirectional(ReLU(64, return_sequences=True)
                      )(x)  # [b_s, seq_len, vec_dim]
    x = Bidirectional(ReLU(64, return_sequences=True)
                      )(x)  # [b_s, seq_len, vec_dim]

    xFirst = Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = Dense(128)(xFirst)

    # dot product attention
    attScores = Dot(axes=[1, 2])([query, x])
    attScores = Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = Dense(64, activation='relu')(attVector)
    x = Dense(32)(x)

    output = Dense(number_of_classes, activation='softmax', name='output')(x)

    model = Model(inputs=[input], outputs=[output])
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


models = {
    'cnn_trad_fpool3': cnn_trad_fpool3,
    'cnn_rnn_attention': cnn_rnn_attention
}
