from scipy.io import wavfile
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
from scipy.io import wavfile
import numpy as np
from pathlib import Path, PurePath
import pickle



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
    ms = fs/1000
    frame_length = frame_length*ms
    frame_stride = frame_stride*ms       
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_stride))  
    frames = []
    for i in range(0, len(data), frame_step):
        frame = data[i: i+frame_length]
        if frame.shape[0] < frame_length:
            frame = np.append(frame, 
                      np.zeros(frame_length - frame.shape[0]))
        frames.append(frame)
    return fs, np.concatenate(frames)                           

def hamming(frames, sample_rate, frame_length=25):  
    print(frames.shape)
    return frames * np.hamming(frame_length)

def power_spectrum(frames, NFFT=512):
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  
    return ((1.0 / NFFT) * ((mag_frames) ** 2)) 


def frequency_to_mel(f, const=1125):
    return (const * np.log10(1 + (f / 2) / 700)) 

def mel_to_frequency(m, const=1125):
    return (700 * (10**(m / const) - 1))  

def filter_banks(pow_frames, sample_rate, NFFT, nfilt):    
    hz_points = mel_to_frequency(
            np.linspace(0, 
            frequency_to_mel(sample_rate),
         nfilt + 2)) 
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    return 20 * np.log10(filter_banks)  # dB



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
    ms = fs/1000
    frame_length = frame_length*ms
    frame_stride = frame_stride*ms       
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_stride))  
    frames = []
    for i in range(0, len(data), frame_step):
        frame = data[i: i+frame_length]
        if frame.shape[0] < frame_length:
            frame = np.append(frame, 
                      np.zeros(frame_length - frame.shape[0]))
        frames.append(frame)
    return fs, np.vstack(frames)                           

def hamming(frames):  
    return frames * np.hamming(frames.shape[1])

def power_spectrum(frames, NFFT=512):
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  
    return ((1.0 / NFFT) * ((mag_frames) ** 2)) 


def frequency_to_mel(f, const=1125):
    return (const * np.log10(1 + (f / 2) / 700)) 


def mel_to_frequency(m, const=1125):
    return (700 * (10**(m / const) - 1))  


def filter_banks(pow_frames, sample_rate, NFFT, nfilt):    
    hz_points = mel_to_frequency(
            np.linspace(0, 
            frequency_to_mel(sample_rate),
         nfilt + 2)) 
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks =  20 * np.log10(filter_banks) 
    
    num_ceps = 12
    cep_lifter = 22
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift 
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    return mfcc

def extract_features(file):    
    sample_rate, frames = read_wav(file)
    frames = hamming(frames)
    pow_frames = power_spectrum(frames)
    frames = filter_banks(pow_frames, sample_rate, 512, 40)
    return frames
        
        

