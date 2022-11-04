### This function i to makeing a poission generator

# fr: is the considered firing rate
# duration: duration of the produced pulses
import numpy as np
import random


def produce_pulses(fr, total_length, time_interval):
    random.seed()
    fr_time = 1.0/fr
    num = int(total_length/time_interval)
    
    pulses = np.zeros(num, dtype=np.float)
    
    k = 0
    inv_interval = 1/time_interval
    while k < num:
        ran = random.random()
        diff = -fr_time*np.log(ran)*inv_interval
        k += int(diff)
        if k < num:
            pulses[k]=pulses[k]+1.0
            
    return pulses

