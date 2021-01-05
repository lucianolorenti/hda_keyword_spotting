import argparse
import logging
import pickle
import time
from pathlib import Path

import mlflow
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
    dataset = Dataset(Path('/home/lucianolorenti/data').resolve())

    current_run = mlflow.start_run()
    mlflow.log_param("train", config['train'])
    mlflow.log_param("model", config['model'])
    mlflow.set_tags({
        'estimator_name': 'cnn_trad_fpool3'
    })
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
                        epochs=epochs,
                        callbacks=[check_point, early_stopping])

    loss = history.history['loss']
    for i, n in enumerate(loss):
        mlflow.log_metric('train_loss', v, step=i)

    loss = history.history['val_loss']
    for i, n in enumerate(loss):
        mlflow.log_metric('val_loss', v, step=i)

    mlflow.log_artifact(model_filename)
