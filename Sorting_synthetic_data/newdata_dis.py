
# This function calculates the distance between the new datapoint and data64 and also calculates the distance threshold 

from math import sqrt
import numpy as np
from OSM import run_OSM_PCA
import matplotlib.pyplot as plt
    
def newdata_dis(GEMsort_C, GEMsort_w, N_all, onedata_new, nodes_total1, path):
    
    #determining th1 (the threshold) 
    Tdim = 64
    GEMsort_metric = 2
    GEMsort_D = np.zeros((GEMsort_w.shape[0], GEMsort_w.shape[0]))
    #GEMsort_crit = []

    for i in range(GEMsort_w.shape[0]):
        for j in range(GEMsort_w.shape[0]):
            if GEMsort_C[i,j] != -1:
                GEMsort_D[i, j] = np.linalg.norm((GEMsort_w[j, :] - GEMsort_w[i, :]), GEMsort_metric) ** 2
    a = GEMsort_D[np.nonzero(GEMsort_D)]
    th1 = np.mean(a)      #th1 is the mean distances of all nodes             

    #Getting PCA of one new data and nodes
    res2 = np.concatenate((N_all, np.reshape(np.array(onedata_new), (1,Tdim))), axis=0)
    X2 = res2 #data with 64 length in every row
    #y = label #labels: 0, 1, 2
    n_components = 2
    M,W,Ys  = run_OSM_PCA(X2, n_components)
    F = (np.linalg.pinv(np.eye(n_components) + M[:n_components ,:n_components ]).dot(W[:n_components ,:])).T
    X_osmpca = (X2 - np.mean(X2, 0)).dot(F)
    X_osmpca = Ys
    #U_new = F.T #eigenvectors

    ### Therefore the last pca element of X_osmpca is related to the new data
    #############################################################################################  
    # Measuring distances of nodes and new pca data and determining the group
    # X_osmpca[i,:] are nodes positions
    N2 = []
    pca_new = X_osmpca[N_all.shape[0],:]

    dis = []
    for i in range(N_all.shape[0]):
        dis.append((np.linalg.norm((pca_new -X_osmpca[i,:]), 2) ** 2))  
    minval, G = np.array(dis).min(), np.array(dis).argmin()
    line_nodes = np.hstack(nodes_total1)

    if G < line_nodes.shape[0]:
        for j in range(len(nodes_total1)):
            for k in range(len(nodes_total1[j])):
                print('G',G)
                np.save(path, line_nodes)
                print(line_nodes)

                if nodes_total1[j][k]==line_nodes[G]:
                    group_node=nodes_total1[j]
                    Group_num = j + 1 

    #num_chosen.append(num_row)#for deleting part 
    ################################################################
    #calculating the th1 (the threshold of the distance for the new data and other data64 points)
        diss = []
        for i in range(group_node.shape[0]):
            for k in range(group_node.shape[0]):
                if i < k:
                    diss.append((np.linalg.norm((X_osmpca[k,:] - X_osmpca[i,:]), 2)))
        th1 = np.mean(diss)

        #X1 = N 
        if minval< th1:
            Group = Group_num  
            nodes_total2 = nodes_total1
            nodess = []  #means there is no new nodes
            dis2 = []
            newclust_finished = 1

        else:
            Group = []
    return(Group,minval,th1,X_osmpca)
