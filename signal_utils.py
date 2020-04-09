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
        numframes = 1 + int(math.floor((1.0 * slen - Nw) / Ns))
    
    if(padding):
        padlen = int((numframes - 1) * Ns + Nw)
        zeros = np.zeros((padlen - slen,))
        audio = np.concatenate((audio, zeros))
    win = window(Nw)
    frames = rolling_window(audio, window=Nw, step=Ns)
    """ NOTE: Author's code returned col wise.... this func returns row-wise """
    return frames * win

def preprocess(audio, buckets, sr=16000, Ws=25, Ss=10, alpha=0.97):
    
    Nw=round((Ws*sr)/1000)
    Ns=round((Ss*sr)/1000)
    
    """Hardcoded for now, coz even author's code gives out 512x299 
    matrix while the paper specifies 512x300. Maybe i'm doing something wrong?"""
    
    Ns=159
    window=np.hamming
    nfft=1<<(Nw-1).bit_length()
    
    audio=rm_dc_n_dither(audio)
    
    audio=preemphasis(audio, alpha)
    
    audio=vec2frames(audio, Nw, Ns, window, False)
    
    mag=abs(fft(audio, nfft))
    
    mag=zscore(mag).T
    
    rsize=max(i for i in buckets if i<=mag.shape[1])
    rstart=(mag.shape[1]-rsize)//2
    
    return mag[:,rstart:rstart+rsize]


if __name__=="__main__":
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
    pp=preprocess(audio[:48000], buckets)
    print(pp.shape)
    