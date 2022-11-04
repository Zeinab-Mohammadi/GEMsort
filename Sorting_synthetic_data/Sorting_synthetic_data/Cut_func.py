
#This is the Cut function to extract spikes

import numpy as np

def cut(x,p,T1): 
    positions =p
    before = int(T1/2)
    after =int((T1/2)-1)
    res = np.zeros((1,(before+after+1)*1)) #
    idx = np.arange(-before,after+1)
    cut = np.zeros((before+after+1))

    keep = idx + positions
    within = np.bitwise_and(0 <= keep, keep < x.shape[0])
    kw = keep[within]
    kw =kw.astype(int) 
    cut = x[kw].copy()
    res[0,0:cut.shape[0]] = cut
    #res=cut
    return res

