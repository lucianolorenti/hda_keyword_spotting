
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import (Activation, Add,
                                            BatchNormalization, Conv2D, Dense,
                                            Embedding, Flatten,
                                            GlobalAveragePooling1D,
                                            GlobalAveragePooling2D, Input,
                                            Lambda, Layer, LayerNormalization,
                                            MultiHeadAttention)
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.ops import math_ops


def ExpandDimension():
    return Lambda(lambda x: K.expand_dims(x))


def cnn_residual_increasing_filters(
    input_shape, number_of_classes, learning_rate=0.001, n_filters=32, n_residuals=3
):

    """
    Deep residual learning for small-footprint keyword spotting.
    Tang, Raphael, and Jimmy Lin.
    """
    input = Input(shape=input_shape)
    x = input

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = ExpandDimension()(x)

    i = 1
    x = Conv2D(
        filters=n_filters,
        strides=(1, 1),
        kernel_size=(3, 3),
        padding="valid",
        use_bias=True,
    )(x)
    x = AveragePooling2D((2, 2))(x)
    i = -1
    for j in range(n_residuals):
        i += 1
        original_x = x
        x = Conv2D(
            filters=n_filters,
            strides=(1, 1),
            kernel_size=(3, 3),
            #dilation_rate=int(2 ** (i // 3)),
            padding="same",
            use_bias=False,
        )(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        i += 1
        x = Conv2D(
            filters=n_filters,
            strides=(1, 1),
            kernel_size=(3, 3),
            #dilation_rate=int(2 ** (i // 3)),
            padding="same",
            use_bias=False,
        )(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)

        x = Add()([original_x, x])
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)

    output = Dense(
        number_of_classes,
        activation="softmax",
        name="output",
        kernel_initializer="he_normal",
    )(x)

    model = Model(inputs=[input], outputs=[output])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        metrics=["accuracy"],
        loss="sparse_categorical_crossentropy",
    )
    return model




class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.class_emb = self.add_weight(
            "class_emb",
            shape=(1, 1, projection_dim)
        )
        self.projection_dim = projection_dim
        self.position_embedding = Embedding(
            input_dim=num_patches+1, output_dim=projection_dim
        )

    def call(self, patch):
        batch_size = tf.shape(patch)[0]
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.projection_dim]
        )

        encoded = tf.concat(
            [class_emb, self.projection(patch)], axis=1
        ) + self.position_embedding(positions)
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




def cnn_visiontransformer(
    input_shape,
    number_of_classes,
    learning_rate=0.001,
    n_dense_layer=1,
    num_heads=3,
    transformer_layers=4,
    projection_dim=16,
    patch_x=8,
    patch_y=8
):
    # implements the Vision Transformer (ViT) model by Alexey Dosovitskiy et al.
    # for image classification

    patch_size = (
        patch_x,
        patch_y,
    )  # Size of the patches to be extract from the input images
    num_patches = int(
        (input_shape[0] / patch_size[0]) * (input_shape[1] / patch_size[1])
    )

    transformer_units = [
        projection_dim,
    ]  # Size of the transformer layers

    input = Input(shape=input_shape)
    x = input
    x = ExpandDimension()(x)
    patches = Patches(patch_size)(x)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):

        x1 = BatchNormalization()(encoded_patches)

        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim
        )(x1, x1)

        x2 = Add()([attention_output, encoded_patches])

        x3 = BatchNormalization()(x2)

        for units in transformer_units:
            x3 = Dense(units, activation=tf.nn.gelu)(x3)


        encoded_patches = Add()([x3, x2])
    representation = encoded_patches

    if np.prod(representation.shape[1:]) > 400:    
        representation = GlobalAveragePooling1D()(representation)
    representation = LayerNormalization(epsilon=1e-6)(representation)
    representation = Flatten()(representation)

    x = representation
    for units in range(n_dense_layer):
        x = Dense(max(int(x.shape[1] / 2),number_of_classes), activation=tf.nn.gelu)(x)

    x = Dense(number_of_classes, activation="softmax")(x)
    model = Model(inputs=[input], outputs=[x])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        metrics=["accuracy"],
        loss="sparse_categorical_crossentropy",
    )
    return model


models = {
    "cnn_residual2": cnn_residual_increasing_filters,
    "cnn_visiontransformer": cnn_visiontransformer,
}
