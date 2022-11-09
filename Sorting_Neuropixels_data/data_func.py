import numpy as np
import scipy, os
import matplotlib.pyplot as plt
import struct
import copy



def get_chunk(mm,start,end,channels,sampling_rate):
	chunk = mm[int(start * sampling_rate * int(channels)):int(np.floor(end * sampling_rate * (int(channels))))]
	#print np.shape(chunk)
	return np.reshape(chunk,(int(channels),-1),order='F') * 0.195



def cut(x,p,T1): 
    positions = p
    before = int(T1 / 2)
    after = int((T1 / 2) - 1)
    res = np.zeros((1,(before + after + 1) * 1)) 
    idx = np.arange(-before, after + 1)
    cut = np.zeros((before + after + 1))

    keep = idx + positions
    within = np.bitwise_and(0 <= keep, keep < x.shape[0])
    kw = keep[within]
    kw = kw.astype(int) 
    cut = x[kw].copy()
    res[0, 0:cut.shape[0]] = cut
    #res = cut
    return res