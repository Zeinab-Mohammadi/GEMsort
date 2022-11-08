########## This is the GEMsort function for clustering the data ##########

import math
import random
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import apply_along_axis as apply
from scipy.signal import fftconvolve
import time

def GEMsort(Nodesnum, GEMsort_dataset):
    
   #########  GEMsort algorithm  #########
    start_time = time.time()
    #k = np.zeros((2, 2))#np.array([np.mean(GEMsort_dataset,axis=0),np.mean(GEMsort_dataset,axis=0)])
    k = np.zeros((2, 2))
    center = k
    GEMsortNumnodes = Nodesnum
    GEMsort_dim = 2  

    #Update learning rate coefficients
    GEMsort_eb = 0.05  # Learning rate of the nearest node 
    GEMsort_en = 0.001  # Learning rate of the neighbors of S 
    GEMsort_beta = 0.001
    GEMsort_alpha = 0.5
    GEMsort_met = 2.
    GEMsort_thr_edge = 5 # Edge threshold for removing inappropriate edges

    init = np.zeros((center.shape[0], 1)) 

    GEMsort_C = -1. + np.diag(init[:, 0])  
    # GEMsort_C is the connection set in which -1 corresponds to no connection

    GEMsort_E = np.zeros((1, center.shape[0])) # Local accumulated error 

    GEMsort_train_node = 10 # Number of training for a new node 
    GEMsort_finish = 0.01

    GEMsort_w = center.copy()

    notinitialsign = 0
    finishsign = 1
    mdl_all = []
    mdl_all_pre = 99999.

    while np.logical_and((GEMsort_w.shape[0] <= GEMsortNumnodes), (finishsign == 1)):
        #print('Training when the number of the nodes in GEMsort is: ' + str(GEMsort_w.shape[0]) + " ...")
        GEMsort_previous = copy.copy(GEMsort_w)
        flag = 1

        for iter2 in range(GEMsort_train_node):
            currentdata = copy.copy(GEMsort_dataset)  
            if flag == 1:
                while (np.array(currentdata).shape[0] - 1) != 0:
                    index = math.ceil(np.array(currentdata).shape[
                                          0] * random.random())  
                    vect =copy.copy(currentdata[np.int(index - 1.), 0:GEMsort_dim])


                    #Determining the winner S1 and the second nearest node S2
                    d = []
                    for i in range(GEMsort_w.shape[0]):
                        d.append((np.linalg.norm((vect - GEMsort_w[i, :]), GEMsort_met) ** 2))  # Finding the error

                    dinit = copy.copy(d)
                    minval, S1 = np.array(dinit).min(), np.array(dinit).argmin()
                    dinit[S1] = 9999.
                    secminval, S2 = np.array(dinit).min(), np.array(dinit).argmin()

                    GEMsort_C[S1, S2] = 0.
                    GEMsort_C[S2, S1] = 0.

                    # S1 Local Error 
                    GEMsort_E[0, S1] += minval  
                    GEMsort_w[S1, :] += GEMsort_eb * (vect - GEMsort_w[S1, :])  # change winner node S1
                    for i in range(GEMsort_C.shape[0]):  
                        if GEMsort_C[S1, i] != -1.0:
                            GEMsort_w[i, :] += GEMsort_en * (vect - GEMsort_w[i, :])

                            
                    for i in range(GEMsort_C.shape[0]):  # Updating the neighbours of the winning node s 
                        if GEMsort_C[S1, i] != -1.0:
                            GEMsort_C[S1, i] += 1.
                            GEMsort_C[i, S1] += 1.

                     # Removal of nodes
                    if notinitialsign == 1:
                        for i in range(GEMsort_C.shape[0]):
                            for j in range(GEMsort_C.shape[1]):
                                 if (GEMsort_C[i, j]) > GEMsort_thr_edge:
                                      GEMsort_C[i, j] = -1.
                                      GEMsort_C[j, i] = -1.

                    
                        for i in range(GEMsort_C.shape[0]):
                            if np.all(GEMsort_C[i, :] == -1.):
                                np.delete(GEMsort_C, i, 0)
                                np.delete(GEMsort_C, i, 1)
                                np.delete(GEMsort_w, i, 0)
                                np.delete(GEMsort_previous, i, 0)
                                np.delete(GEMsort_E, i, 1)

                    # Reducing error of all nodes
                    GEMsort_E = GEMsort_E * (1. - GEMsort_beta)

                    currentdata = np.delete(currentdata, index-1, 0)  # Delete the used samples

                    crit = 0
                    for i in range(GEMsort_w.shape[0]):
                        crit += np.linalg.norm(GEMsort_previous[i, :] - GEMsort_w[i, :], GEMsort_met)

                    crit = crit / GEMsort_w.shape[0]
                    if crit <= GEMsort_finish:
                        #print"stop"
                        flag = 0
                    else:
                        GEMsort_previous = copy.copy(GEMsort_w)
                        
        # Adding new node
        if GEMsort_w.shape[0] < GEMsortNumnodes:
            init2 = copy.copy(GEMsort_E)
            maxval, q = init2.max(), init2.argmax()
            init2[0, q] = -99999999.0
            secmaxval, p = init2.max(), init2.argmax()

            # Finding node with maximum error
            f = []
            maxval = -99999999.
            for i in range(np.array(GEMsort_C).shape[0]):
                if GEMsort_C[q, i] != -1.:
                    if GEMsort_E[0, i] > maxval:
                        maxval = GEMsort_E[0, i].copy()
                        f = i

            # Insert first new node r1
            if f!=[]:
                a = (GEMsort_w[q, :] + GEMsort_w[f, :]) / 2.  
                GEMsort_w = np.vstack((GEMsort_w, a))
            
                r1 = GEMsort_w.shape[0]  
        
                p = (-1) * np.ones((np.array(GEMsort_C).shape[0], 1))
                GEMsort_C = np.concatenate((GEMsort_C, p), axis=1)
                t = (-1) * np.ones((1, np.array(GEMsort_C).shape[1]))
                GEMsort_C = np.concatenate((GEMsort_C, t), axis=0)

                GEMsort_C[q, (r1 - 1)] = 0  # Making new connections
                GEMsort_C[(r1 - 1), q] = 0
                GEMsort_C[f, (r1 - 1)] = 0
                GEMsort_C[(r1 - 1), f] = 0
                GEMsort_C[q, f] = -1.  # deleting the conncections between nodes (q and f)
                GEMsort_C[f, q] = -1.

            # Using GEMsort_alpha to decrease local error variables of q and f 
                GEMsort_E[0, q] = (1. - GEMsort_alpha) * GEMsort_E[0, q]
                GEMsort_E[0, f] = (1. - GEMsort_alpha) * GEMsort_E[0, f]

       
                E_r = (GEMsort_E[0, q] + GEMsort_E[0, f]) / 2.
                GEMsort_E = np.append(GEMsort_E, E_r)
                GEMsort_E = np.reshape(GEMsort_E, (1, GEMsort_E.shape[0]))

        GEMsort_D = np.zeros((GEMsort_w.shape[0], GEMsort_w.shape[0]))
        GEMsort_crit=[]

        for i in range(GEMsort_w.shape[0]):
            for j in range(GEMsort_w.shape[0]):
                GEMsort_D[i, j] = np.linalg.norm((GEMsort_w[j, :] - GEMsort_w[i, :]), GEMsort_met) ** 2


        if GEMsort_w.shape[0] == GEMsortNumnodes:
            finishsign = 0

        notinitialsign = 1


    GEMsort_C[np.logical_and((GEMsort_C != -1), (GEMsort_D > np.mean(GEMsort_D)))] = -1


    # GEMsort_NewC=[]
    #
    #
    # for i in range(GEMsort_C.shape[0]):
    #     for j in range (GEMsort_C.shape[0]):
    #        if GEMsort_C[i,j] != -1:
    #            GEMsort_NewC.append(GEMsort_C[i,j])
    #
    # for i in range(GEMsort_C.shape[0]):
    #     for j in range (GEMsort_C.shape[0]):
    #        if GEMsort_C[i,j] > np.mean(GEMsort_NewC):
    #           GEMsort_C[i, j] = -1
    #

    dis_mean = np.zeros((GEMsort_C.shape[0],GEMsort_C.shape[0]))
    dis = np.zeros((1,GEMsort_dataset.shape[0]))

    for i in range(GEMsort_C.shape[0]):
        for j in range(GEMsort_C.shape[0]):
             if GEMsort_C[i, j] != -1.:
                 for k in range(GEMsort_dataset.shape[0]):
                     midx = (GEMsort_w[j, 0] + GEMsort_w[i, 0]) / 2.
                     midy = (GEMsort_w[j, 1] + GEMsort_w[i, 1]) / 2.
                     dis[0, k] = np.sqrt(((midx - GEMsort_dataset[k, 0]) * (midx - GEMsort_dataset[k, 0])) + ((midy - GEMsort_dataset[k, 1]) * (midy - GEMsort_dataset[k, 1])))
                 num_dis = np.sort(dis)[:5]
                 dis_mean[i,j] = np.mean(num_dis)

    
    dis_mean = np.array(dis_mean)
    tt = (dis_mean[dis_mean != 0].shape)
    sum_total = np.sum(dis_mean[dis_mean != 0])
    mean_total = sum_total/tt


    for i in range(dis_mean.shape[0]):
        for j in range(dis_mean.shape[1]):
            if dis_mean[i, j] != 0:
                if dis_mean[i,j] > (1.2 * mean_total):
                    GEMsort_C[i, j] = -1
                    GEMsort_C[j, i] = -1

    #determinig number of groups

    all_nodes=[]
    set1 = []
    set2 = []
    for i in range(Nodesnum):
        for j in range(Nodesnum):
            if GEMsort_C[i, j] != -1.:
                set1.append(i)
                set2.append(j)

    d = np.zeros((np.array(set1).shape[0], 2))
    d[:, 0] = set1
    d[:, 1] = set2

    for k in range(Nodesnum):
        nodes = []
        for i in range(d.shape[0]):
            nodes.append(k)
            if d[i, 0] == k:
                nodes.append(d[i, 1])

            # print nodes
        for i in range(d.shape[0]):
            for j in nodes:
                if d[i, 0] == j:
                    nodes.append(d[i, 1])

        for i in range(d.shape[0]):
            if d[i, 1] == k:
                nodes.append(d[i, 0])
        for i in range(d.shape[0]):
            for j in (nodes):
                if d[i, 1] == j:
                    nodes.append(d[i, 0])
        nodes = np.unique(nodes)
        all_nodes.append(nodes)
   
    #identifying final groups
    nodes_total = []

    if nodes.shape[0]!=0:
        for k in range(Nodesnum):
            a = []
            for j in range(Nodesnum):
                if np.array(all_nodes)[j][-1] == k:
                    a = np.array(all_nodes)[j]
            if a != []:
                nodess=a
                nodes_total.append(a)
        nodes_total 

        Nodes = []
        for i in range(GEMsort_C.shape[0]):
            if i < Nodesnum:
                for j in range (i, GEMsort_C.shape[0]):
                    if GEMsort_C[i,j] != -1:
                        x = [GEMsort_w[i,0],GEMsort_w[j,0]]
                        y = [GEMsort_w[i, 1], GEMsort_w[j, 1]]
                        Nodes.append(GEMsort_w[i,:])
                        Nodes.append(GEMsort_w[j,:])
        Nodes = np.unique(np.array(Nodes), axis=0)
        
    if nodes.shape[0] == 0:
        Nodes = []  
        nodess = []
        
#     plt.plot(GEMsort_dataset[:,0],GEMsort_dataset[:,1], 'o', zorder=1)
#     for i in range(GEMsort_C.shape[0]):
#         for j in range (GEMsort_C.shape[0]):
#             if GEMsort_C[i,j] != -1:
#                # print i,j
#                 x = [GEMsort_w[i,0], GEMsort_w[j,0]]
#                 y = [GEMsort_w[i, 1], GEMsort_w[j, 1]]
#                 plt.plot(x, y, 'k', zorder=1, lw=1)  # 'b' color of lines, lw:width of lines
#                 plt.scatter(x, y, s=120, c='darkorange', zorder=2)
#     plt.show()
    return Nodes,GEMsort_C,GEMsort_w,nodes_total,Nodesnum,d,nodess

