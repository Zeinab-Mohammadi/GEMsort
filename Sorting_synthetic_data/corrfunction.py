import numpy as np
from Cut_func import cut
import scipy.signal
import matplotlib.pyplot as plt

# T1: length of each spike
# ch_num: number of channels
# T: considered length of data
# Tp: threshold for peak function 
# Tdis: Peaks' minimum distance threshold for peak function

def corr(X, ch_num, T1 , Tp, Tdis,T):
    rowp=[]
    p_all = np.array([], dtype = object).reshape((1,0))
    #stage 1: detecting peaks of all channels

    for i in range (X.shape[0]): 
        x = X[i,0:T] 
        p,_ = scipy.signal.find_peaks(x, height=Tp, distance=Tdis) 
        #print(i)
        rowp.append(p) 
        #print(rowp)

        if p==[]:
            p=[0] #to prevent error, so in every row if p=0 means: no peak in that channel
        y_arr = np.array([], dtype=np.int32)
        y=p
        y_arr = np.append(y_arr,y)
        #print(y_arr)
        p_all = np.append(p_all, 0)
        #print(p_all)
        p_all[-1] = y_arr.astype(int)   

    a=np.hstack(rowp) 
    rowp=np.unique(a)[None,:]
    rowp=rowp.astype(int)
    rows=[]
    signals=np.zeros((np.array(rowp).shape[1],ch_num,T1)) 

    #stage 2: finding the similar peaks in nearby times in near channels
    for k in range (np.array(rowp).shape[1]): 
        rows.append([])

    for k in range (np.array(rowp).shape[1]): #for every peak check 
        for i in range (p_all.shape[0]): 
            for j in range (p_all[i].shape[0]):
                if (p_all[i][j]!=0 and np.abs(p_all[i][j]-rowp[0,k])<2 and X[i,rowp[0,k]]!=0): 
                    rows[k].append(i) # for all the channels number with the same peak(k)
                    x = X[i,0:T]
                    signals[k,i,:]=cut(x,p_all[i][j],T1)  #this will contain all 64 signals with those peaks

    old_sig=np.copy(signals)

    th_cor=0.5
    maxx=np.zeros((1,np.array(rowp).shape[1]))
    zeross=0 #this is the total number of peaks that became zero
    for k in range (np.array(rowp).shape[1]): 

        if (rows[k]!=[] and np.array(rows[k]).shape[0]!=1) : 
            maxx[0,k]=-999999
            for p in range (np.array(rows[k])[:,None].shape[0]) : #considering number of same peaks in every group           
                for t in range (np.array(rows[k])[:,None].shape[0]):
                    
                    if (rows[k][p]!=rows[k][t] and np.abs(rows[k][p]-rows[k][t])<3 and np.any(signals[k,rows[k][p],:])!=0 and np.any(signals[k,rows[k][t],:])!=0):                 
                        corr=np.corrcoef(signals[k,rows[k][p],:], signals[k,rows[k][t],:])[0, 1]
                        if corr >= th_cor: # if corrolation is more than the threshold
                            if np.abs(X[rows[k][p],rowp[0,k]]) >= maxx[0,k]:
                                maxx[0,k]=np.abs(X[rows[k][p],rowp[0,k]])
                                #print(maxx[0,k])
                            else:    
                                signals[k,rows[k][p],:]=np.zeros((1,T1)) #if that signal dose not include the max, remove it 
                                zeross=zeross+1
                                #print('rowp', rows[k][p]) #numer of channel which is removed
                            if np.abs(X[rows[k][t],rowp[0,k]]) > maxx[0,k]: #X[...] is the amount of signal in peak k
                                maxx[0,k]=np.abs(X[rows[k][t],rowp[0,k]])
                                signals[k,rows[k][p],:]=np.zeros((1,T1))
#                                 print('rowp', rows[k][p])
#                                 print(maxx[0,k])
                                zeross=zeross+1
                            else:    
                                signals[k,rows[k][t],:]=np.zeros((1,T1))
                                zeross=zeross+1

#now removing zero signals and put them in one array for getting pca                                    

    num=np.sum(np.any(signals, axis=2))    
    res=np.zeros((num,T1)) 
    chnum=np.zeros((num,1)) 
    p=0
    for i in range (signals.shape[0]):
        for j in range (signals.shape[1]):
            if (np.any(signals[i,j,:])!=0):
                res[p,:]=signals[i,j,:]
                chnum[p,0]=j
                p=p+1   
                
    return res,chnum,zeross                            

    
