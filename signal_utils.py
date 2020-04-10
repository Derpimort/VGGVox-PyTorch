#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:32:02 2020

@author: darp_lord
"""
import numpy as np
from numpy.fft import fft
import math
from scipy.signal import lfilter
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

# Following 2 functions derived from
# https://github.com/jameslyons/python_speech_features
def rolling_window(a, window, step=1):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]

def vec2frames(audio, Nw, Ns, window, padding=False):
    slen = len(audio)
    if slen <= Nw:
        numframes = 1
    else:
        #Changed to math.floor to ensure consistency between implementations
        numframes = 1 + int(math.floor((1.0 * slen - Nw) / Ns))
    
    #Default padding check
    if(padding):
        padlen = int((numframes - 1) * Ns + Nw)
        zeros = np.zeros((padlen - slen,))
        audio = np.concatenate((audio, zeros))
    #No need to do anything if padding=False, strides func takes care of truncation
    win = window(Nw)
    frames = rolling_window(audio, window=Nw, step=Ns)
    
    return frames * win


def normalize_frames(m,epsilon=1e-12):
	return np.array([(v - np.mean(v)) / max(np.std(v),epsilon) for v in m])

def preprocess(audio, buckets=None, sr=16000, Ws=25, Ss=9.9375, alpha=0.97):
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
    
    
    """Hardcoded for now, coz even author's code gives out 512x299 
    matrix while the paper specifies 512x300. Maybe i'm doing something wrong?"""
    
    #Ns=159
    #hamming window func signature
    window=np.hamming
    #get next power of 2 greater than or equal to current Nw
    nfft=1<<(Nw-1).bit_length()
    
    # Remove DC and add small dither
    audio=rm_dc_n_dither(audio)
    
    # Preemphasis filtering
    audio=preemphasis(audio, alpha)
    
    #Get 400x300 frames with hamming window
    audio=vec2frames(audio, Nw, Ns, window, False)
    
    #get 512x300 spectrograms... zscore is just mean and variance normalization.
    mag=abs(fft(audio, nfft)) 
    mag=normalize_frames(mag.T)
    #mag=zscore(mag).T
    
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
    