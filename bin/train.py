import tensorflow as tf
from keyword_spotting.data import (Batcher, Iterator, RawDataset,
                                   generate_keras_batcher)
from keyword_spotting.model import get_model, get_model_2, get_model_tcn

dataset = RawDataset('/home/lucianolorenti/data')


train_batcher, val_batcher, a, b, input_shape = generate_keras_batcher(
    dataset,  512, True)

#
#
# callbacks = [
# tf.keras.callbacks.TensorBoard(
# log_dir='/home/luciano/tblogs',
# update_freq='batch')
# ]
model = get_model_2(input_shape, dataset.number_of_classes)
model.summary()
model.fit(train_batcher, epochs=50, validation_data=val_batcher,
          steps_per_epoch=len(a),
          validation_steps=len(b))
