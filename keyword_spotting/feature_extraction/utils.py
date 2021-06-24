import pickle
from pathlib import Path, PurePath

import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
from scipy.io import wavfile
from python_speech_features import mfcc

def read_wav(file, frame_length=25, frame_stride=10):
    """
    Read the given file and return window

    Arguments
    ---------
    frame_length: int.
                  Length of every frame in ms

    frame_stride: int
                  Window separation in ms

    Returns
    -------
    np.array([])
    """
    fs, data = wavfile.read(file)
    return np.int64(fs), np.float32(data)




def extract_features(signal, sample_rate, NFFT=512, nfilt=40):
    return mfcc(signal, sample_rate, numcep=40, nfilt=nfilt, nfft=NFFT).astype('float32')


def extract_features_padding(signal, sample_rate, NFFT=512, nfilt=40):
    a =  mfcc(signal, sample_rate, numcep=40, nfilt=nfilt, nfft=NFFT).astype('float32')
    b = np.zeros((100, 40), dtype=np.float32)
    b[: a.shape[0], :] = a
    return b

def windowed(data, label, left: int = 30, right: int = 10):
    d = np.array(
        [data[j - left : j + right, :] for j in range(left, data.shape[0] - right)],
        dtype=np.float32,
    )
    labels = np.array(
        [label for j in range(left, data.shape[0] - right)], dtype=np.int32
    )

    return d, labels


def dct(n_filters, n_input):
    basis = np.empty((n_filters, n_input))
    basis[0, :] = 1.0 / np.sqrt(n_input)

    samples = np.arange(1, 2 * n_input, 2) * np.pi / (2.0 * n_input)

    for i in range(1, n_filters):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / n_input)

    return basis


dct_filters = dct(40, 40)
