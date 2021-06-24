from keyword_spotting.model import PatchEncoder, Patches
import pickle

from numpy import random
import simpleaudio as sa
from simpleaudio.shiny import WaveObject
from keyword_spotting.feature_extraction.utils import read_wav
import numpy as np
from scipy.io.wavfile import write
import pyaudio
import wave
from pathlib import Path
import tensorflow as tf

NOISE_FILES = [
    "doing_the_dishes.wav",
    "dude_miaowing.wav",
    "exercise_bike.wav",
    "pink_noise.wav",
    "running_tap.wav",
    "white_noise.wav",
]


def _write_recorded_audio(
    filename: Path, p, frames, sample_format, channels: int, fs: int
):
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b"".join(frames))
    wf.close()


def record_audio(filename: Path):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 16000
    seconds = 2

    p = pyaudio.PyAudio()

    stream = p.open(
        format=sample_format,
        channels=channels,
        rate=fs,
        frames_per_buffer=chunk,
        input=True,
    )

    frames = []

    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    p.terminate()
    _write_recorded_audio(filename, p, frames, sample_format, channels, fs)


def play_audio_from_file(filename: Path):
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()


def add_noise(
    sample_rate,
    signal,
    dataset_path: Path,
    random_noise: bool = True,
    noise_factor: float = 0.1,
):
    if (random_noise and np.random.rand() > 0.8) or not random_noise:
        noise_path = dataset_path / "_background_noise_" / np.random.choice(NOISE_FILES)
        fs, data_noise = read_wav(noise_path)
        min_length = min(data_noise.shape[0], signal.shape[0])
        noise_factor = np.random.rand() * noise_factor
        signal[:min_length] = (
            signal[:min_length] + noise_factor * data_noise[:min_length]
        )
    return sample_rate, signal


def play_audio_from_array(
    signal: np.ndarray, sample_rate: int = 16000, num_channels: int = 1
):
    wave_obj = WaveObject(
        signal.astype(np.int16), num_channels=num_channels, sample_rate=sample_rate
    )
    wave_obj.play()


def load_model(model_path: str):
    h5_file = Path(model_path + ".h5")
    if h5_file.is_file():
        model_path = str(h5_file)
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder},
    )


def mROC(predictions, ytrues):
    r = []
    words = np.unique(ytrues)
    for w in words:
        FPR = []
        TPR = []
        FNR = []
        word_y = ytrues == w
        P = sum(word_y)
        N = len(word_y) - P
        for thresh in np.linspace(0, 1, 100):
            FP = 0
            TP = 0
            FN = 0
            for i in range(predictions.shape[0]):
                if predictions[i, w] >= thresh:
                    if word_y[i] == 1:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if word_y[i] == 1:
                        FN += 1
            FPR.append(FP / N)
            TPR.append(TP / P)
            FNR.append(FN / P)
        r.append((np.array(FPR), np.array(TPR), np.array(FNR)))
    return r


def average_ROC_curves(r, N:int = 500):
    AA = np.linspace(0, 0.2, N + 1)
    BB = np.zeros(N)
    for i in range(len(r)):
        FPR, TPR, FNR = r[i]
        for i in range(len(AA) - 1):
            BB[i] += np.mean(FNR[(FPR >= AA[i]) & (FPR >= AA[i + 1])])
    BB[i] /= len(r)
    return AA, BB