
#This is the smoothing function

import copy
import numpy as np
from scipy.signal import butter, lfilter

def Smoothing_filter(data_raw, low, high, sf, order):
    # Determine Nyquist frequency
    nyq_freq = sf / 2

    # Set bands
    low = low / nyq_freq
    high = high / nyq_freq

    # Calculate coefficients
    b, a = butter(order, [low, high], btype='band')

    # Filter signal
    filtered_data = lfilter(b, a, data_raw)

    return filtered_data

