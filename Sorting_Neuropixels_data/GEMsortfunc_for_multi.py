 #########  GEMsort algorithm  ###################

import math
import random
#from numpy import *
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import apply_along_axis as apply
from scipy.signal import fftconvolve
import time


def GEMsort(prenumnode, GEMsort_dataset,GEMsort_eb,GEMsort_en,GEMsort_edgethred,GEMsort_beta,GEMsort_alpha):
    start_time = time.time()

    # k=np.zeros((2,2))  #np.array([np.mean(GEMsort_dataset,axis=0),np.mean(GEMsort_dataset,axis=0)])
    k = np.zeros((2,2))
    center = k

    # Predefined Number of nodes
    GEMsort_Pre_numnode = prenumnode
    GEMsort_NoFeats = 2  
    GEMsort_metric = 2.
    temp = np.zeros((center.shape[0], 1))  
    GEMsort_C = -1. + np.diag(temp[:, 0]) #'-1' represents having no connection

    # GEMsort_C is connection set
    GEMsort_E = np.zeros((1, center.shape[0]))

    # Training epoch for every new inserted node
    GEMsort_epochspernode = 10
    GEMsort_stopcriteria = 0.01

    # Initializing reference vector weights
    GEMsort_w = center.copy()

    # Training the network
    nofirsttimeflag = 0
    stopflag = 1
    allmdlvalue = []
    previousmdlvalue = 99999.
    # t = 0;

    while np.logical_and((GEMsort_w.shape[0] <= GEMsort_Pre_numnode), (stopflag == 1)):
        #print('Training when the number of the nodes in GEMsort is: ' + str(GEMsort_w.shape[0]) + " ...")
        GEMsort_previousw = copy.copy(GEMsort_w)
        flag = 1

        for iter2 in range(GEMsort_epochspernode):
            workdata = copy.copy(GEMsort_dataset)  # Copy to working dataset from which used samples are removed
            if flag == 1:
                while (np.array(workdata).shape[0] - 1) != 0:
                    index = math.ceil(np.array(workdata).shape[
                                          0] * random.random())  # Choose a training sample randomly from the training dataset
                    CurVec = copy.copy(workdata[np.int(index - 1.), 0:GEMsort_NoFeats])


                    d = []
                    for i in range(GEMsort_w.shape[0]):
                        d.append((np.linalg.norm((CurVec - GEMsort_w[i, :]), GEMsort_metric) ** 2))  # Find squared error

                    dtemp = copy.copy(d)
                    minval, S1 = np.array(dtemp).min(), np.array(dtemp).argmin()
                    dtemp[S1] = 9999.
                    secminval, S2 = np.array(dtemp).min(), np.array(dtemp).argmin()

                    GEMsort_C[S1, S2] = 0.
                    GEMsort_C[S2, S1] = 0.

                    GEMsort_E[0, S1] += minval 

                    GEMsort_w[S1, :] += GEMsort_eb * (CurVec - GEMsort_w[S1, :])  # Update winner node S1
                    for i in range(GEMsort_C.shape[0]):  # find neighbors of the winning node s and update them
                        if GEMsort_C[S1, i] != -1.0:
                            GEMsort_w[i, :] += GEMsort_en * (CurVec - GEMsort_w[i, :])

                    # Increase the age of all edges emanating from S1
                    for i in range(GEMsort_C.shape[0]):  # Find neighbours of the winning node s and update them
                        if GEMsort_C[S1, i] != -1.0:
                            GEMsort_C[S1, i] += 1.
                            GEMsort_C[i, S1] += 1.


                    # Removing nodes
                    if nofirsttimeflag == 1:
                        for i in range(GEMsort_C.shape[0]):
                            for j in range(GEMsort_C.shape[1]):
                                 if (GEMsort_C[i, j]) > GEMsort_edgethred:
                                      #maxval, Sr = np.array(GEMsort_C[i, :]).max(), np.array(GEMsort_C[i, :]).argmax()
                                      GEMsort_C[i, j] = -1.
                                      GEMsort_C[j, i] = -1.

                        # temp = []
                        for i in range(GEMsort_C.shape[0]):
                            if np.all(GEMsort_C[i, :] == -1.):
                                np.delete(GEMsort_C, i, 0)
                                np.delete(GEMsort_C, i, 1)
                                np.delete(GEMsort_w, i, 0)
                                np.delete(GEMsort_previousw, i, 0)
                                np.delete(GEMsort_E, i, 1)


                                #               t=t+1;
                                #               if mod(t,50) == 0,
                                #                  hold off
                                #                  #plot(GEMsort_dataset(1:size(GEMsort_dataset,1)-4,1),GEMsort_dataset(1:size(GEMsort_dataset,1)-4,2),'r.')
                                #                  hold on
                                #                  plot(GEMsort_w(:,1),GEMsort_w(:,2),'bx')
                                #                  drawnow
                                #

                    # Decrease error of all units
                    GEMsort_E = GEMsort_E * (1. - GEMsort_beta)

                    workdata = np.delete(workdata, index-1, 0)  # Remove the used samples

                    crit = 0
                    for i in range(GEMsort_w.shape[0]):
                        crit += np.linalg.norm(GEMsort_previousw[i, :] - GEMsort_w[i, :], GEMsort_metric)

                    crit = crit / GEMsort_w.shape[0]
                    if crit <= GEMsort_stopcriteria:
                        #print"stop"
                        flag = 0
                    else:
                        GEMsort_previousw = copy.copy(GEMsort_w)

                      
        if GEMsort_w.shape[0] < GEMsort_Pre_numnode:
            temp2 = copy.copy(GEMsort_E)
            maxval, q = temp2.max(), temp2.argmax()
            temp2[0, q] = -99999999.0
            secmaxval, p = temp2.max(), temp2.argmax()

            f = []
            maxval = -99999999.
            for i in range(np.array(GEMsort_C).shape[0]):
                if GEMsort_C[q, i] != -1.:
                    if GEMsort_E[0, i] > maxval:
                        maxval = GEMsort_E[0, i].copy()
                        f = i

            if f == []:
                print('Unable to find a link to split for node insertion')

            # Insert first new node r1
            if f != []:
                a = (GEMsort_w[q, :] + GEMsort_w[f, :]) / 2.  # Add the new reference vector for new node r1 from nodes q and f
                GEMsort_w = np.vstack((GEMsort_w, a))
            # GEMsort_w = [GEMsort_w;(GEMsort_w[q, :] + GEMsort_w[f,:]) / 2.]# Add the new reference vector for new node r1 from nodes q and f
                r1 = GEMsort_w.shape[0]  # Find index of r1
            # GEMsort_C = [GEMsort_C, (-1) * np.ones((np.array(GEMsort_C).shape[0], 1))]# Expand C
            # GEMsort_C = [GEMsort_C, (-1) * np.ones((1, np.array(GEMsort_C).shape[1]))]
                p = (-1) * np.ones((np.array(GEMsort_C).shape[0], 1))
                GEMsort_C = np.concatenate((GEMsort_C, p), axis=1)
                t = (-1) * np.ones((1, np.array(GEMsort_C).shape[1]))
                GEMsort_C = np.concatenate((GEMsort_C, t), axis=0)

                GEMsort_C[q, (r1 - 1)] = 0  # Insert new connections
                GEMsort_C[(r1 - 1), q] = 0
                GEMsort_C[f, (r1 - 1)] = 0
                GEMsort_C[(r1 - 1), f] = 0
                GEMsort_C[q, f] = -1.  # Remove original conncections between q and f
                GEMsort_C[f, q] = -1.

            # Decrease local error variables of q and f by a fraction GEMsort_alpha
                GEMsort_E[0, q] = (1. - GEMsort_alpha) * GEMsort_E[0, q]
                GEMsort_E[0, f] = (1. - GEMsort_alpha) * GEMsort_E[0, f]

                E_r = (GEMsort_E[0, q] + GEMsort_E[0, f]) / 2.
                GEMsort_E = np.append(GEMsort_E, E_r)
                GEMsort_E = np.reshape(GEMsort_E, (1, GEMsort_E.shape[0]))
            # ([GEMsort_E, (GEMsort_E[q] + GEMsort_E[f]) / 2.])

        GEMsort_D = np.zeros((GEMsort_w.shape[0], GEMsort_w.shape[0]))
        GEMsort_crit = []

        for i in range(GEMsort_w.shape[0]):
            for j in range(GEMsort_w.shape[0]):
                GEMsort_D[i, j] = np.linalg.norm((GEMsort_w[j, :] - GEMsort_w[i, :]), GEMsort_metric) ** 2


        # if GEMsort_C.shape==GEMsort_D.shape :
        #     for i in range(GEMsort_C.shape[0]):
        #          for j in range(GEMsort_C.shape[0]):
        #               if GEMsort_C[i, j] != -1:
        #                   if GEMsort_D[i, j] > np.mean(GEMsort_D):
        #                       GEMsort_C[i, j] = -1
        #                       GEMsort_C[j, i] = -1
        #print GEMsort_C 

        if GEMsort_w.shape[0] == GEMsort_Pre_numnode:
            stopflag = 0


        nofirsttimeflag = 1


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
    #           GEMsort_C[i, j]=-1
    #


    dis_mean = np.zeros((GEMsort_C.shape[0],GEMsort_C.shape[0]))
    dis = np.zeros((1,GEMsort_dataset.shape[0]))

    for i in range(GEMsort_C.shape[0]):
        for j in range(GEMsort_C.shape[0]):
             if GEMsort_C[i, j] != -1.:
                 for k in range(GEMsort_dataset.shape[0]):
                     #A=(GEMsort_w[i,1]-GEMsort_w[j,1])/(GEMsort_w[j,0]-GEMsort_w[i,0])
                     #B=1.
                     #C=(-A*GEMsort_w[i,0])-GEMsort_w[i,1]
                     #dis[0,k]=np.abs(A*GEMsort_dataset[k,0]+B*GEMsort_dataset[k,1]+C)/(np.sqrt((A*A)+(B*B)))
                     midx = (GEMsort_w[j, 0] + GEMsort_w[i, 0]) / 2.
                     midy = (GEMsort_w[j, 1] + GEMsort_w[i, 1]) / 2.
                     dis[0, k] = np.sqrt(((midx - GEMsort_dataset[k, 0]) * (midx - GEMsort_dataset[k, 0])) + (
                     (midy - GEMsort_dataset[k, 1]) * (midy - GEMsort_dataset[k, 1])))
                 num_dis = np.sort(dis)[:5]
                 dis_mean[i,j] = np.mean(num_dis)

    dis_mean = np.array(dis_mean)
    tt = (dis_mean[dis_mean != 0].shape)
    sum_total = np.sum(dis_mean[dis_mean != 0])
    mean_total = sum_total / tt


    for i in range(dis_mean.shape[0]):
        for j in range(dis_mean.shape[1]):
            if dis_mean[i, j] != 0:
                if dis_mean[i,j] > (1.2 * mean_total):
                    GEMsort_C[i, j] = -1
                    GEMsort_C[j, i] = -1


    all_nodes = []
    set1 = []
    set2 = []
    for i in range(prenumnode):
        for j in range(prenumnode):
            if GEMsort_C[i, j] != -1.:
                set1.append(i)
                set2.append(j)

    d = np.zeros((np.array(set1).shape[0], 2))
    d[:, 0] = set1
    d[:, 1] = set2

    for k in range(prenumnode):
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
    nodes_total = []     #identifying final groups


    if nodes.shape[0] != 0:
        for k in range(prenumnode):
            a = []
            for j in range(prenumnode):
                if np.array(all_nodes)[j][-1] == k:
                    a = np.array(all_nodes)[j]
            if a != []:
                nodess = a
                nodes_total.append(a)
        nodes_total 

        Nodes = []
        for i in range(GEMsort_C.shape[0]):
            if i < prenumnode:
                for j in range (i,GEMsort_C.shape[0]):
                    if GEMsort_C[i,j] != -1:
                        x = [GEMsort_w[i,0],GEMsort_w[j,0]]
                        y = [GEMsort_w[i, 1], GEMsort_w[j, 1]]
                        Nodes.append(GEMsort_w[i,:])
                        Nodes.append(GEMsort_w[j,:])
        Nodes = np.unique(np.array(Nodes), axis=0)
        
    if nodes.shape[0] == 0:
        Nodes = []  
        nodess = []
        
    #plt.plot(GEMsort_dataset[:,0],GEMsort_dataset[:,1], 'o', zorder=1)
    #for i in range(GEMsort_C.shape[0]):
        #for j in range (GEMsort_C.shape[0]):
            #if GEMsort_C[i,j] != -1:
               # print i,j
                #x=[GEMsort_w[i,0],GEMsort_w[j,0]]
                #y=[GEMsort_w[i, 1], GEMsort_w[j, 1]]
                #plt.plot(x, y, 'k', zorder=1, lw=1)  
                #plt.scatter(x, y, s=120, c='darkorange', zorder=2)
    #plt.show()
    return Nodes, GEMsort_C, GEMsort_w, nodes_total, prenumnode, d, nodess

