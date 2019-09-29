#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 14:30:44 2018

@author: Ananya Banerjee
"""

import numpy as np
import pandas as pd
import scipy.spatial.distance as sc
import operator

leaf_data = np.loadtxt('leaf.data',dtype=np.dtype(float),delimiter=',')

k_points = [12,18,24,36,42]

#function to get data as we want it
def get_data(dataset):
    data = dataset[:,1:]
    classes = dataset[:,0]
    
    return data,classes

#function to standardize data
def mean_var(data):
    m,n = data.shape
    
    for i in range(n):
        avg = np.mean(data[:,i])
        std = np.std(data[:,i])
        data[:,i] = (data[:,i] - avg)/float(std)
        
    return data

#function that returns eucledian distance between x and y
def euclidean_distance(x,y):
    return sc.euclidean(x,y)


#function to assign clusters
def assign_to_cluster(data,means,k,cluster):
    for i in range(len(data)):
        min_dist = 1000
        index = -1
        for j in range(len(means)):
            dist = euclidean_distance(data[i],means[j])
            if dist<min_dist:
                min_dist = dist
                index = j
        cluster[index].append(i)
    
    return cluster

#function to update cluster means
def update_cluster_means(data,means,k,cluster):
    avg = 0.0
    for i in range(k):
        mu = 0.0
        for j in range(len(cluster[i])):
            #print "cluster length",cluster[i]
            data_point = cluster[i][j]
            mu = mu + data_point
            
        if len(cluster[i])!=0:
            avg = mu/float(len(cluster[i]))


        means[i] = avg
    return means
    
#function for k-means algorithm 
def kmeans(data,k):
    m,n = data.shape
    cl=[]
    for i in range(20):
        means = []
        cluster = []
        flag = 0
        for j in range(k):
            mu = np.random.uniform(-3,3)
            means.append(mu)
            cluster.append([])
        #set means update points
        
        while flag!=1:
            prev_cluster = cluster
            cluster = assign_to_cluster(data,means,k,cluster)
            #print "cluster :",cluster
            means = update_cluster_means(data,means,k,cluster)
            if cluster==prev_cluster:
                flag = 1
        #set points update means
        print means
        cl.append(cluster)
    return means, cl
   
#function to extract datapoints from cluster
def extract_datapoints(cluster, data, k):
    l=[[] for i in range(k)]
    #print "cluster is", cluster, len(cluster[0]),k
    for j in range(len(cluster)):
        #cluster[0] will be a list of data points
        li=cluster[j]
        #print "list is", li
        l[j]=data[li, :]
    
    print l
    return l

#function to calculate value of objectice function
def objective_func(means,cluster,data,k):
    for n in range(20):
        #l contains the list of lists of data points in an cluster each
        l=extract_datapoints(cluster[n], data, k)
        obj_val=0.0
        for i in range(k):
            for j in range(len(l)):
                list_1=l[j]
                l2=[x-means[i] for x in list_1]
                norm=np.linalg.norm(l2)
                sqr=norm**2
                obj_val+=sqr
                
        obj_val=float(obj_val)/float(data.shape[0])
    obj_val=float(obj_val)/float(20)   
    return obj_val
            
        
#function to extract mean and var correspodning to each k
def ext():
    l=[]
    for i in list(K_M.keys()):
        t=K_M[i]
        l.append([t[2],t[3]])
        
    return l
     
#main
data,classes = get_data(leaf_data)
data = mean_var(data)

#dictionary to store k as key and mean and var as value list
K_M=dict()

for k in k_points:
    means, cluster=kmeans(data,k)
    obj_val=objective_func(means,cluster,data,k)
    var=float(np.sum(np.power((data-obj_val),2)))/float(data.shape[0])
    K_M[k]=[means,cluster, obj_val, var]
    
l=ext()  
print "list of mean and var is"l