from keyword_spotting.data import Dataset,  DataGenerator
from keyword_spotting.model import get_model, get_model_2
import tensorflow as tf 


dataset = Dataset('/home/luciano/speech/')
train_generator = DataGenerator(dataset, 'train')
validation_generator = DataGenerator(dataset, 'validation')
callbacks = [tf.keras.callbacks.TensorBoard(
    log_dir='/home/luciano/tblogs',
    update_freq='batch')]
model = get_model_2((31, dataset.input_shape[1], 1), dataset.number_of_classes())
model.fit(train_generator, callbacks=callbacks, epochs=50, validation_data=validation_generator)

store_test(model, preprocessing, result)