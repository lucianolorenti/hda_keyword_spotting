import tensorflow as tf
from tcn import TCN
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import (GRU, Activation, Add,
                                            AveragePooling2D,
                                            BatchNormalization, Bidirectional,
                                            Concatenate, Conv2D, Dense, Dot,
                                            Dropout, Flatten, Input, Lambda,
                                            LayerNormalization, Reshape,
                                            Softmax, SpatialDropout2D, 
                                            Embedding, Layer, MultiHeadAttention,
                                            GlobalAveragePooling2D, AveragePooling1D)

from tensorflow.python.ops import math_ops
import numpy as np
from tensorflow import keras
import tensorflow_addons as tfa

def ExpandDimension():
    return Lambda(lambda x: K.expand_dims(x))


def cnn_residual_increasing_filters(input_shape,  number_of_classes, learning_rate=0.001, n_filters=32, n_residuals=3):
    """
    Deep residual learning for small-footprint keyword spotting.
    Tang, Raphael, and Jimmy Lin. 
    """
    input = Input(shape=input_shape)
    x = input

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = ExpandDimension()(x)

    i= 1
    x = Conv2D(filters=n_filters,
            strides=(1, 1),
            kernel_size=(3, 3),
            #dilation_rate=int(2**(i // 3)),
            padding='valid',
            use_bias=True)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    for j in range(n_residuals):
        original_x = x 
        x = Conv2D(filters=n_filters,
                   strides=(1, 1),
                   kernel_size=(3, 3),
                   dilation_rate=int(2**(j // 3)),
                   padding='same',
                   use_bias=False)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=n_filters,
                   strides=(1, 1),
                   kernel_size=(3, 3),
                   dilation_rate=int(2**(j // 3)),
                   padding='same',
                   use_bias=False)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        x = Add()([original_x, x])


    x = Conv2D(filters=n_filters,
               strides=(1, 1),
               kernel_size=(3, 3),
               padding='valid',
               use_bias=True)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n_filters,
               strides=(1, 1),
               kernel_size=(3, 3),
               padding='valid',
               use_bias=True)(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output = Dense(number_of_classes, activation='softmax', name='output', kernel_initializer='he_normal')(x)

    model = Model(inputs=[input], outputs=[output])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
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


def cnn_inception(input_shape, number_of_classes, n_filters=16, sizes=[5, 10, 15]):
    """
    Convolutional Neural Networks for Small-footprint Keyword Spotting
    Tara N. Sainath, Carolina Parada
    """
    input = Input(shape=input_shape)
    x = input
    x = ExpandDimension()(x)

    x1 = Conv2D(n_filters,
                strides=(1, 3),
                kernel_size=(sizes[0], 16),
                padding='same',
                activation='relu',
                use_bias=False)(x)

    x2 = Conv2D(n_filters,
                strides=(1, 3),
                kernel_size=(sizes[1], 8),
                padding='same',
                activation='relu',
                use_bias=False)(x)

    x3 = Conv2D(n_filters,
                strides=(1, 3),
                kernel_size=(sizes[2], 4),
                padding='same',
                activation='relu',
                use_bias=False)(x)
    x = Concatenate()([x1, x2, x3])

    x = Conv2D(64,
               strides=(1, 1),
               kernel_size=(10, 4),
               padding='same',
               use_bias=False)(x)

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
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')
    return model


def cnn_inception2(input_shape, number_of_classes, learnig_rate=0.001, n_filters=16, sizes=[5, 10, 15]):
    """
    Convolutional Neural Networks for Small-footprint Keyword Spotting
    Tara N. Sainath, Carolina Parada
    """
    input = Input(shape=input_shape)
    x = input
    x = ExpandDimension()(x)

    x1 = Conv2D(n_filters,
                strides=(1, 3),
                kernel_size=(sizes[0], 16),
                padding='same',
                activation='relu',
                use_bias=False)(x)

    x2 = Conv2D(n_filters,
                strides=(1, 3),
                kernel_size=(sizes[1], 8),
                padding='same',
                activation='relu',
                use_bias=False)(x)

    x3 = Conv2D(n_filters,
                strides=(1, 3),
                kernel_size=(sizes[2], 4),
                padding='same',
                activation='relu',
                use_bias=False)(x)
    x = Concatenate()([x1, x2, x3])

    x1 = Conv2D(n_filters,
                strides=(1, 1),
                kernel_size=(5, 4),
                padding='same',
                activation='relu',
                use_bias=False)(x)

    x2 = Conv2D(n_filters,
                strides=(1, 1),
                kernel_size=(10, 4),
                padding='same',
                activation='relu',
                use_bias=False)(x)

    x3 = Conv2D(n_filters,
                strides=(1, 1),
                kernel_size=(15, 4),
                padding='same',
                activation='relu',
                use_bias=False)(x)

    x = Concatenate()([x1, x2, x3])

    x = Flatten()(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(number_of_classes, activation='softmax')(x)
    model = Model(inputs=[input], outputs=[x])
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
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

class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

def cnn_visiontransformer(input_shape, number_of_classes, learning_rate=0.001):
    #implements the Vision Transformer (ViT) model by Alexey Dosovitskiy et al. 
    # for image classification, and demonstrates it on the CIFAR-100 dataset.
   
    patch_size = (8,8)  # Size of the patches to be extract from the input images
    num_patches = int((input_shape[0] / patch_size[0])*(input_shape[1] / patch_size[1])) 
    projection_dim = 16
    num_heads = 3
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 4
    mlp_head_units = [64]  # Size of the dense layers of the final classifier

    input = Input(shape=input_shape)
    x = input
    x = ExpandDimension()(x)
    # Create patches.
    patches = Patches(patch_size)(x)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation= AveragePooling1D(3)(encoded_patches)
    representation = LayerNormalization(epsilon=1e-6)(representation)
    representation = Flatten()(representation)
    representation = Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.

    x = Dense(number_of_classes, activation='softmax')(features)
    model = Model(inputs=[input], outputs=[x])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
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
    'cnn_residual2': cnn_residual_increasing_filters, 
    'cnn_inception2': cnn_inception2,
    'cnn_visiontransformer':cnn_visiontransformer, 
}
