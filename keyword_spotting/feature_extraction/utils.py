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


def hamming(frames, frame_length):
    return frames * np.hamming(frame_length)


def power_spectrum(frames, NFFT=512):
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    return ((1.0 / NFFT) * ((mag_frames) ** 2))


def frequency_to_mel(f, const=2595):
    return (const * np.log10(1 + (f / 2) / 700))


def mel_to_frequency(m, const=2595):
    return (700 * (10**(m / const) - 1))


def sliding_window(signal, sample_rate, frame_size=0.025, frame_stride=0.01):
    frame_length, frame_step = (frame_size * sample_rate, frame_stride *
                                sample_rate)
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    num_frames = int(np.ceil(
        float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))

    pad_signal = np.append(signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    return frames, frame_length


def filter_banks(pow_frames, sample_rate, NFFT, nfilt, num_ceps: int = 12, cep_lifter: int = 22):
    hz_points = mel_to_frequency(
        np.linspace(0,
                    frequency_to_mel(sample_rate),
                    nfilt + 2))
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))), dtype=np.float32)

    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)

    filter_banks = np.where(filter_banks == 0, np.finfo(
        float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)

    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')
    mfcc = mfcc[:, 1: (num_ceps + 1)]

    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    return mfcc


def open_extract_features(file, NFFT=256, nfilt=40):
    sample_rate, signal = read_wav(file)
    return extract_features(sample_rate, signal, NFFT=NFFT, nfilt=nfilt)
    


def extract_features(signal, sample_rate, NFFT=256, nfilt=40):
    return mfcc(signal, sample_rate, numcep=40, nfilt=nfilt, nfft=512).astype('float32')
    #frames, frame_length = sliding_window(
    #    signal, sample_rate)
    #frames = power_spectrum(hamming(frames, frame_length), NFFT=NFFT)
    #frames = filter_banks(frames,
    #                          sample_rate,
    #                          NFFT,
    #                          nfilt)#

    #return np.float32(frames)


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
