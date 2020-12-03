from pathlib import Path

import numpy as np
import tensorflow as tf
import pickle
from tqdm.auto import tqdm



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

    def number_of_classes(self):
        return len(self.labels_unique)

    def preproces(self, folder):
        pass

    def full_path(self, file_name):
        return self.folder / file_name


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset, what, batch_size=128,  shuffle=True):
        'Initialization'
        self.dataset = dataset
        if what == 'train':
            self.files = self.dataset.training_files
            self.labels = self.dataset.training_labels
        elif what == 'validation':
            self.files = self.dataset.validation_files
            self.labels = self.dataset.validation_labels
        elif what == 'test':
            self.files = self.dataset.testing_files
            self.labels = self.dataset.testing_labels

        self.indices = list(range(1, len(self.labels)))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Number of batches per epoch"
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indices)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, indices):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = []
        y = []
        # Generate data
        for index in indices:
            file_path = self.dataset.full_path(self.files[index])
            with open(file_path, 'rb') as file:
                frames = pickle.load(file)
                left = 23
                right = 8
                window_width = left+right
                for i in range(left, frames.shape[0]-right):
                    sliding = frames[i-left:i+right]
                    if sliding.shape[0] < window_width:
                        sliding = np.hstack(sliding, np.zeros(
                            window_width-sliding.shape[0], sliding.shape[1]))
                    X.append(sliding)
                    y.append(self.labels[index])
        
        X = np.expand_dims(np.array(X), 3)
        y = np.array(y)
        return X, y
