import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import (Input,Dropout, Lambda, Flatten, Dense, Conv2D, BatchNormalization)
from tcn import TCN
from tensorflow.keras import Model

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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')
    return model

def ExpandDimension():
    return Lambda(lambda x: K.expand_dims(x))


def get_model_2(input_shape, number_of_classes):
    input = Input(shape=input_shape)
    x = input
    x = ExpandDimension()(x)

    x = Conv2D(16,
                kernel_size=(3, 7),
                activation='relu',
                padding='same',
                use_bias=True)(x)
    x = BatchNormalization()(x)        
    x = Dropout(0.2)(x)
    x = Flatten()(x)   
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x) 
    x = Dropout(0.2)(x)
    x = Dense(number_of_classes, activation='softmax')(x)
    model =  Model(inputs=[input], outputs=[x])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=0.2, clipvalue=0.5)
    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')
    return model

def get_model_tcn(input_shape, number_of_classes):
    input = Input(shape=input_shape)
    x = input
    x = TCN(32,use_skip_connections=True, dropout_rate=0.3, use_batch_norm=True)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(number_of_classes, activation='softmax')(x)
    model =  Model(inputs=[input], outputs=[x])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,  clipnorm=0.001, clipvalue=0.001)
    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')
    return model

