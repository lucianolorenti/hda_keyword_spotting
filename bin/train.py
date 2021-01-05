import argparse
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from keyword_spotting.data import Dataset
from keyword_spotting.model import cnn_trad_fpool3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='Config file')

    args = parser.parse_args()

    config = None
    with open(args.config, 'r') as file:
        config = yaml.load(file.read(), Loader=yaml.SafeLoader)
    dataset = Dataset('/home/luciano/speech')
    dataset.to_numpy()
    batch_size = config['train']['batch_size']
    train_data, val_data, test_data = dataset.get_sequences(
        batch_size=batch_size)

    model = cnn_trad_fpool3(dataset.shape, dataset.number_of_classes)

    epochs = config['train']['epochs']

    model_path = Path(config['model']['path']).resolve()
    model_filename = model_path / \
        (config['model']['name'] + time.strftime("%Y%m%d_%H%M%S"))

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)
    check_point = tf.keras.callbacks.ModelCheckpoint(model_filename)
    history = model.fit(train_data,
                        validation_data=val_data,
                        epochs=epochs)

    with open(model_filename + '_history.pkl', 'wb') as file:
        pickle.dump(history, file)
