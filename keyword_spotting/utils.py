import os

import librosa
import numpy as np
from tqdm.auto import tqdm


def WAV2Numpy(folder, sr=None):
    """
    Recursively converts WAV to numpy arrays.
    Deletes the WAV files in the process

    folder - folder to convert.
    """
    allFiles = []
    for root, dirs, files in os.walk(folder):
        allFiles += [os.path.join(root, f) for f in files
                     if f.endswith('.wav')]

    for file in tqdm(allFiles):
        y, sr = librosa.load(file, sr=None)
        np.save(file + '.npy', y)
        os.remove(file)


def moving_average(data_set, periods=3):
    weights = np.ones(periods) / (periods)
    return np.convolve(data_set, weights, mode='valid')


def moving_average_1(data, w):
    b = np.abs(np.random.randn(50)*np.random.rand(50)*500)
    b = b / np.sum(b)
    return [np.mean(b[max(0, j-w):j]) for j in range(b.shape[0])]


def confidence(probas, w, j):
    for i in len(probas):
        h = max(0, j-w)
        np.maximum(probas[h, j])
    return [np.prod(b[max(0, j-w):j]) for j in range(b.shape[0])]
