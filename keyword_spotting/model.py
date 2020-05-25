import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K

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


def get_model_2(input_shape, number_of_classes):
    model = tf.keras.models.Sequential([
     
        tf.keras.layers.Conv2D(16,
                               strides=(1, 3),
                               kernel_size=(3, 7),
                               activation='relu',
                               input_shape=input_shape,
                               padding='same',
                               use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(number_of_classes, activation='softmax')
    ])
 
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')
    return model


class Model:
    def __init__(self, input_shape, number_of_classes):
        self.model = get_model_2(input_shape, number_of_classes)

    def predict(self, what):
        predictions = [self.model.predict(f) for f in frames(what)]
