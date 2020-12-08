import pickle
from pathlib import Path
from typing import Union

import tensorflow as tf
from keyword_spotting.data import TransformedDataset
from keyword_spotting.model import get_model, get_model_2, get_model_tcn

dataset = TransformedDataset('/home/luciano/speech2')
model = get_model_2(dataset.shape, dataset.number_of_classes)
train_data, val_data, test_data = dataset.get_iterators()
model.fit(train_data.batch(250),
          validation_data=val_data.batch(250), epochs=15)
