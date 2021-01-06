import argparse
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
import yaml
from keyword_spotting.data import TransformedDataset
from keyword_spotting.model import models

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
        Path(config['dataset']['path']).resolve(), suffix='basic')
    current_run = mlflow.start_run()
    mlflow.log_param("train", config['train'])
    mlflow.log_param("model", config['model'])

    timestampStr = time.strftime("%Y%m%d_%H%M%S")

    model_id = config['model']['name'] + '_' + timestampStr
    mlflow.set_tags({
        'estimator_type':  config['model']['name']
    })

    batch_size = config['train']['batch_size']
    train_data, val_data, test_data = dataset.get_sequences(
        batch_size=batch_size)

    model = models['cnn_trad_fpool3'](dataset.shape, dataset.number_of_classes)
    model.summary()
    epochs = config['train']['epochs']

    model_path = Path(config['model']['path']).resolve()
    model_filename = model_path / '{model_id}.hf5'

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)
    check_point = tf.keras.callbacks.ModelCheckpoint(model_filename)
    history = model.fit(train_data,
                        validation_data=val_data,
                        epochs=epochs,
                        callbacks=[check_point, early_stopping])

    loss = history.history['loss']
    for i, v in enumerate(loss):
        mlflow.log_metric('train_loss', v, step=i)

    loss = history.history['val_loss']
    for i, v in enumerate(loss):
        mlflow.log_metric('val_loss', v, step=i)

    mlflow.log_artifact(model_filename)

    predicted = model.predict(test_data)

    predicted_path = model_path / '{model_id}_predictions.pkl'
    with open(predicted_path, 'wb') as file:
        pickle.dump(predicted, file)

    mlflow.log_artifact(predicted_path)
