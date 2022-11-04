
### finding nearest 64-length waveforms of spikes to GEMsort nodes
from math import sqrt
import numpy as np
from OSM import run_OSM_PCA
import matplotlib.pyplot as plt
from numpy.linalg import svd

def neardata64(Nodes,new_dataset,spikedata):
    NN=np.zeros((Nodes.shape[0],2))
    N=np.zeros((Nodes.shape[0],64))
    
    for i in range(Nodes.shape[0]):
        dis3=[]
        for j in range(new_dataset.shape[0]):
            dis3.append(np.abs(np.sum((Nodes[i,:] - new_dataset[j,:]))))
            
        minval3, G3 = np.array(dis3).min(), np.array(dis3).argmin()
        N[i,:]=spikedata[G3,:]  # this is the array containing original data (64-length data)
        NN[i,:]=new_dataset[G3,:] 

    res=N
    varcovmat = np.cov(res.T)
    k=np.array(np.mean(res.T,1))

    tt=[]
    for i in range(res.shape[0]):
        tt.append(k)
    varcovmat=res.T-np.array(tt).T

    u, s, v = svd(varcovmat)
    X_osmpca0=np.dot(res,u[:,0:2])
#     plt.scatter(PCA_initial[:,0],PCA_initial[:,1])
    
#     X1 = N 
#     n_components =2
#     M,W,Ys  = run_OSM_PCA(X1,n_components)
#     F=(np.linalg.pinv(np.eye(n_components)+M[:n_components ,:n_components ]).dot(W[:n_components ,:])).T
#     X_osmpca0 = (X1-np.mean(X1,0)).dot(F)
#     X_osmpca0 = Ys
#     U_new=F.T 

    #### ploting#####
    plt.close('all')
    plt.figure()   
    plt.scatter(X_osmpca0[:, 0], X_osmpca0[:, 1])
    #plt.scatter(X_osmpca0[0:N.shape[0], 0], X_osmpca0[0:N.shape[0], 1], c='b')
    #plt.scatter(X_osmpca0[30, 0], X_osmpca0[30, 1], c='g')        
    plt.legend(loc="best")
    #plt.axis([-5, 4, -2, 3])

    plt.show()
    return(N)

