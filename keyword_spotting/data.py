from pathlib import Path
from typing import Union
import numpy as np
import tensorflow as tf
import pickle
from tqdm.auto import tqdm
from sklearn.utils import shuffle
import math 



class LRUDataCache:
    def __init__(self, max_elem):
        self.max_elem = max_elem
        self.data = {}

    def get(self, key):
        self.data[key]['hit'] += 1

        return self.data[key]['elem']

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
    def __init__(self, folder):
        self.folder = Path(folder)

        files = [str(f.relative_to(self.folder))
                 for f in self.folder.glob('**/*.wav')]

        self.testing_files = read_lines(self.folder/'testing_list.txt')
        self.validation_files = read_lines(self.folder/'validation_list.txt')[:500]

        self.training_files = list(
            set(files) - set(self.testing_files) - set(self.validation_files))[:5000]

        change_extension(self.training_files, '.pkl')
        change_extension(self.testing_files, '.pkl')
        change_extension(self.validation_files, '.pkl')

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

        with open(self.full_path(self.training_files[0]), 'rb') as file:
            self.input_shape = pickle.load(file).shape

    @property
    def number_of_classes(self):
        return len(self.labels_unique)

    def preproces(self, folder):
        pass

    def full_path(self, file_name):
        return self.folder / file_name


class Iterator:
    def __init__(self,
                 dataset: Dataset,
                 what: str,
                 shuffle: Union[bool, str] = False):
        self.dataset = dataset
        if what == 'train':
            files = self.dataset.training_files
            labels = self.dataset.training_labels
        elif what == 'validation':
            files = self.dataset.validation_files
            labels = self.dataset.validation_labels
        elif what == 'test':
            files = self.dataset.testing_files
            labels = self.dataset.testing_labels

        self.cache = LRUDataCache(50)
        self.left = 23
        self.right = 8
        self.elements = self.compute_elements(what, files, labels)
        self.i = 0
        self.shuffle = shuffle


    @property
    def shape(self):
        return self[0][0].shape

    def compute_elements(self, what, files, labels):
        cached_elements_path = self.dataset.folder / f'{what}_elements.pkl'
        if (cached_elements_path).is_file():
            with open(cached_elements_path, 'rb') as file:
                return pickle.load(file)

        elements = []
        for f, label in tqdm(list(zip(files, labels))):
            file_path = self.dataset.full_path(f)
            with open(file_path, 'rb') as file:
                frames = pickle.load(file)
            window_width = self.left+self.right
            for i in range(self.left, frames.shape[0]-self.right):
                elements.append((file_path, label, i))

        with open(cached_elements_path, 'wb') as file:
            pickle.dump(elements, file)
        return elements

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            shuffle(self.elements)
        return self

    def at_end(self):
        return self.i == len(self.elements)

    def __getitem__(self, i: int):       
        return self._load_data(i)

    def __len__(self):
        return len(self.elements)

    def _load_data(self, index):        
        file_path = self.elements[index][0]
        with open(file_path, 'rb') as file:
            frames = pickle.load(file)

        i = self.elements[index][2]
        label = [self.elements[index][1]]
        window_width = self.left+self.right

        sliding = frames[i-self.left:i+self.right]
        if sliding.shape[0] < window_width:
            sliding = np.hstack(sliding, 
            np.zeros(window_width-sliding.shape[0], sliding.shape[1]))
        return (sliding, label)

    def __next__(self):
        if self.at_end():
            raise StopIteration
        ret = self.__getitem__(self.i)
        self.i += 1
        return ret



class Batcher:
    def __init__(self,
                 iterator: Iterator,
                 batch_size: int,
                 restart_at_end: bool = True):
        self.iterator = iterator
        self.batch_size = batch_size
        self.restart_at_end = restart_at_end
        self.stop = False

    def __len__(self):
        return math.ceil(len(self.iterator) / self.batch_size)

    def __iter__(self):
        self.iterator.__iter__()
        return self

    def __next__(self):
        X = []
        y = []
        if self.stop:
            raise StopIteration
        if self.iterator.at_end():
            if self.restart_at_end:
                self.__iter__()
            else:
                raise StopIteration
        try:
            for _ in range(self.batch_size):
                X_t, y_t = next(self.iterator)
                X.append(np.expand_dims(X_t, axis=0))
                y_t = np.expand_dims(y_t, axis=0)
                y.append(y_t)
        except StopIteration:
            pass
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        return X.astype(np.float32), y.astype(np.float32)


def generate_keras_batcher(dataset,  batch_size, shuffle=False):

    it_train = Iterator(dataset, 'train', shuffle)
    b_train = Batcher(it_train, batch_size)

    it_val = Iterator(dataset, 'validation')
    b_val = Batcher(it_val, batch_size)
    shape = it_train.shape

    def gen_train():
        for X, y in b_train:
            yield X, y

    def gen_val():
        for X, y in b_val:
            yield X, y

    a = tf.data.Dataset.from_generator(
        gen_train, (tf.float32, tf.float32), (tf.TensorShape(
            [None, shape[0], shape[1]]), tf.TensorShape([None, 1])))
    b = tf.data.Dataset.from_generator(
        gen_val, (tf.float32, tf.float32), (tf.TensorShape(
            [None, shape[0], shape[1]]), tf.TensorShape([None, 1])))
    return a, b, b_train, b_val, shape