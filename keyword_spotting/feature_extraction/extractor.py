import argparse
import logging
import pickle
from pathlib import Path, PurePath
from typing import Union

import numpy as np
import tensorflow as tf
from keyword_spotting.data import Dataset
from keyword_spotting.feature_extraction.utils import extract_features
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class FeatureExtractor:
    def __init__(self, dataset: Dataset, output_path: Union[str, Path], suffix: str = '', left: int = 23, right: int = 8):
        self.dataset = dataset
        self.output_path = output_path
        self.left = left
        self.right = right
        if len(suffix) > 0:
            self.suffix = '_' + suffix

    def write(self):
        logger.debug('Writing')
        self.nrow, self.ncol = self.write_dataset(
            self.dataset.training_files, 'train')
        self.write_dataset(self.dataset.testing_files, 'test')
        self.write_dataset(self.dataset.validation_files, 'validation')
        self.write_info()

    def write_info(self):
        with open(self.output_path / f'output_info{self.suffix}', 'wb') as file:
            pickle.dump({
                'nrow': self.nrow,
                'ncol': self.ncol,
                'classes': self.dataset.label_to_index,
                'n_classes': self.dataset.number_of_classes
            },
                file)

    def write_dataset(self, files: list, what: str):
        nrow = None
        ncol = None
        print(str(self.output_path / f'output_{what}{self.suffix}'))
        writer = tf.io.TFRecordWriter(
            str(self.output_path / f'output_{what}{self.suffix}'))
        for name in tqdm(files):
            name = self.dataset.full_path(name)
            file_path_parts = name.parts
            data = extract_features(name)
            for j in range(self.left, data.shape[0]-self.right):
                window = data[j-self.left:j+self.right, :]

                if nrow is not None and nrow != window.shape[0]:
                    raise ValueError('Invalid nrow')
                if ncol is not None and ncol != window.shape[1]:
                    raise ValueError('Invalid ncol')
                nrow = window.shape[0]
                ncol = window.shape[1]
                d = {
                    'data': tf.train.Feature(float_list=tf.train.FloatList(value=window.reshape(-1))),
                    'target': _int64_feature(self.dataset.get_class(file_path_parts[-2]))

                }
                d = tf.train.Example(features=tf.train.Features(feature=d))
                writer.write(d.SerializeToString())
        return nrow, ncol
