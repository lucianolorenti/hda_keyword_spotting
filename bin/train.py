import argparse
import logging
import pickle
import time
from pathlib import Path, PurePath
from typing import Union

import numpy as np
import tensorflow as tf
import yaml
from keyword_spotting.data import Dataset, TransformedDataset
from keyword_spotting.feature_extraction.extractor import FeatureExtractor
from keyword_spotting.model import cnn_trad_fpool3, get_model, get_model_tcn
from tqdm.auto import tqdm

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

    dataset = TransformedDataset(
        config['dataset']['path'], config['dataset']['name'])
    model = cnn_trad_fpool3(dataset.shape, 16, dataset.number_of_classes)
    train_data, val_data, test_data = dataset.get_iterators()

    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']

    model_path = Path(config['model']['path']).resolve()
    model_filename = model_path / \
        (config['model']['name'] + time.strftime("%Y%m%d_%H%M%S"))

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=7)
    check_point = tf.keras.callbacks.ModelCheckpoint(model_filename)
    history = model.fit(train_data.batch(batch_size),
                        validation_data=val_data.batch(batch_size),
                        epochs=epochs)

    with open(model_filename + '_history.pkl', 'wb') as file:
        pickle.dump(history, file)
