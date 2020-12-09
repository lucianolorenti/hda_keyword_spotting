import functools
import math
import pickle
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tqdm.auto import tqdm


class LRUDataCache:
    def __init__(self, max_elem):
        self.max_elem = max_elem
        self.data = {}

    def get(self, key):
        self.data[key]['hit'] += 1

        return self.data[key]['elem']

    def __contains__(self, key):
        return key in self.data

    def add(self, key, elem):
        if len(self.data) == self.max_elem:
            keys = list(self.data.keys())
            key_to_remove = np.argmin([self.data[k]['hit'] for k in keys])
            del self.data[keys[key_to_remove]]
        self.data[key] = {
            'elem':  elem,
            'hit': 0
        }
        return elem

    def __len__(self):
        return len(self.data)


def obtain_target(f):
    """
        given the path something/target/target.{pkl.wav} return the label

    Argumnets
    ---------


    Return
    ------
    string: label of the given file
    """
    return Path(f).parts[-2]


def read_lines(f):
    """
    Return all the lines of the file

    Arguments
    ---------
    f: string, or pathlib.Path
       file to be read

    Return
    ------
    list of str
    """
    with open(f, 'r') as file:
        return [line.rstrip() for line in file]


def change_extension(l, ext):
    def change_extension_to(f, ext):
        return f.split('.')[0] + ext
    for i, file in enumerate(l):
        l[i] = change_extension_to(file, ext)


class Dataset:
    def __init__(self, path: Union[Path, str]):
        self.folder = path
        if isinstance(path, str):
            self.folder = Path(path)

        files = [str(f.relative_to(self.folder))
                 for f in self.folder.glob('**/*.wav')]

        self.testing_files = read_lines(self.folder/'testing_list.txt')
        self.validation_files = read_lines(self.folder/'validation_list.txt')

        self.training_files = list(
            set(files) - set(self.testing_files) - set(self.validation_files))

        self.training_labels = [obtain_target(f) for f in self.training_files]
        self.testing_labels = [obtain_target(f) for f in self.testing_files]
        self.validation_labels = [obtain_target(
            f) for f in self.validation_files]

        self.labels_unique = np.unique(self.training_labels)
        self.label_to_index = {label: idx for idx,
                               label in enumerate(self.labels_unique)}
        self.index_to_label = {idx: label for idx,
                               label in enumerate(self.labels_unique)}

        self.training_labels = [self.label_to_index[l]
                                for l in self.training_labels]
        self.testing_labels = [self.label_to_index[l]
                               for l in self.testing_labels]
        self.validation_labels = [self.label_to_index[l]
                                  for l in self.validation_labels]

    @property
    def number_of_classes(self):
        return len(self.labels_unique)

    def full_path(self, file_name):
        return self.folder / file_name

    def get_class(self, class_str):
        return self.label_to_index[class_str]


class RawDataset(Dataset):
    pass


class TransformedDataset:
    def __init__(self, path: Union[Path, str], suffix=''):
        if isinstance(path, str):
            path = Path(path).resolve()
        self.path = path
        self.suffix = suffix
        if len(suffix) > 0:
            self.suffix = '_' + suffix
        self.load_info()
        self.feature_description = {
            'data': tf.io.FixedLenFeature([self.info['nrow']*self.info['ncol']], tf.float32),
            'target': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        }
        self.shape = (self.info['nrow'], self.info['ncol'])

    @property
    def number_of_classes(self):
        return self.info['n_classes']

    def load_info(self):
        with open(self.path / f'output_info{self.suffix}', 'rb') as file:
            self.info = pickle.load(file)

    def generate_dataset(self, what: str):
        def _parse_data(pattern):
            parsed = tf.io.parse_single_example(
                pattern, self.feature_description)
            data = tf.reshape(
                parsed['data'],
                shape=self.shape
            )
            return data, parsed['target']
        dataset = tf.data.TFRecordDataset(
            str(self.path / f'output_{what}{self.suffix}'))
        return dataset.map(_parse_data)

    def get_iterators(self):
        return (
            self.generate_dataset('train'),
            self.generate_dataset('validation'),
            self.generate_dataset('test'),
        )
