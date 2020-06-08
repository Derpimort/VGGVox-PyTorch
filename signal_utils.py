#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:32:02 2020

@author: darp_lord
"""
import numpy as np
from numpy.fft import fft
import math
from scipy.signal import lfilter, stft
from scipy.stats import zscore
from scipy.io import wavfile


def rm_dc_n_dither(audio):
    # All files 16kHz tested..... Will copy for 8kHz from author's matlab code later
    alpha=0.99
    b=[1,-1]
    a=[1,-alpha]
    
    audio=lfilter(b,a,audio)
    
    dither=np.random.uniform(low=-1,high=1, size=audio.shape)
    spow=np.std(audio)
    return audio+(1e-6*spow)*dither

def preemphasis(audio, alpha=0.97):
    b=[1, -alpha]
    a=1
    return lfilter(b, a, audio)

def normalize_frames(m,epsilon=1e-12):
    return (m-m.mean(1, keepdims=True))/np.clip(m.std(1, keepdims=True),epsilon, None)

def preprocess(audio, buckets=None, sr=16000, Ws=25, Ss=10, alpha=0.97):
    #ms to number of frames
    if not buckets:
        buckets={100: 2,
             200: 5,
             300: 8,
             400: 11,
             500: 14,
             600: 17,
             700: 20,
             800: 23,
             900: 27,
             1000: 30}
    
    Nw=round((Ws*sr)/1000)
    Ns=round((Ss*sr)/1000)
    
    
    #hamming window func signature
    window=np.hamming
    #get next power of 2 greater than or equal to current Nw
    nfft=1<<(Nw-1).bit_length()
    
    # Remove DC and add small dither
    audio=rm_dc_n_dither(audio)
    
    # Preemphasis filtering
    audio=preemphasis(audio, alpha)
    
    
    #get 512x300 spectrograms
    _, _, mag=stft(audio,
    fs=sr, 
    window=window(Nw), 
    nperseg=Nw, 
    noverlap=Nw-Ns, 
    nfft=nfft, 
    return_onesided=False, 
    padded=False, 
    boundary=None)

    mag=normalize_frames(np.abs(mag))
    
    #Get the largest bucket smaller than number of column vectors i.e. frames
    rsize=max(i for i in buckets if i<=mag.shape[1])
    rstart=(mag.shape[1]-rsize)//2
    #Return truncated spectrograms
    return mag[:,rstart:rstart+rsize]


if __name__=="__main__":
    # Test file same as one on the authors github for testing and maintaining consistency
    sr, audio=wavfile.read("test.wav")
    buckets={100: 2,
     200: 5,
     300: 8,
     400: 11,
     500: 14,
     600: 17,
     700: 20,
     800: 23,
     900: 27,
     1000: 30}
    print(audio.shape)
    # Crop and pass 3s audio for preprocessing
    pp=preprocess(audio, buckets)
    print(pp.shape)
    
