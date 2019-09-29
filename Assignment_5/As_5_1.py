#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 13:25:27 2018

@author: Ananya Banerjee

"""
import numpy as np
from numpy import linalg as lg
import pandas as pd
import math
from cvxopt import matrix, solvers  #this package helps solve quadratic optimization problems
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

#Accuracy Train: key: k, val"accuracy"
ACCURACY_train=dict()

#Mean dict : with k as key and mean val as value
MEAN=dict()

#Variance dict: with k as key and varaiane of cluster list as value
VARIANCE=dict()

#train dataset
train_file=open("/Users/Mammu/Desktop/courses/machine learning/Assignment_5_ml/leaf.data")
dataset_train=np.loadtxt(train_file, dtype=np.dtype(float), delimiter=',')

df=dataset_train
df=np.delete(df,0,axis=1)
#labels=dataset_train[::-1] choosing column 0
labels=dataset_train[:,0]

#preprocess the data
scaler = StandardScaler().fit(df)
df_rescaled = scaler.transform(df)


#function to calculate accuracy
def find_accuracy(y_true, y_pred):
    count=0
    for i in range(len(y_true)):
        if y_true[i]==y_pred[i]:
            count+=1
    acc=float(count)/float(len(y_true))
    acc=acc*100
    print "Accuracy is", acc
    return acc

#function to give 20 different intializations for each attribute
def initialize(X, k):
     t=np.unique(labels)
     #create a array
     arr=np.zeros((k,X.shape[1]))
     #print "shape os", arr.shape, "t is", t
     #init is (n_clusters, n_features) 
     for i in range(k):
         for j in range(X.shape[1]):
             # range of sampling uniformly: [-3,3]
             s = np.random.uniform(-3,3,1)
             #print "random generated",s
             arr[i][j]=s
             
     
     #print arr.shape
     return arr
     

#function to perform k-means
def perform_k_means(k, X, labels):#, X_test):
    arr=initialize(X,k)
    km=KMeans(n_clusters=k, random_state=0,init=arr)#,random_state=0)
    model=km.fit(X)
    pred_labels=model.labels_
    centroids= model.cluster_centers_
    D=cdist(X, centroids, 'euclidean')
    #print "dist is", D
    sum_1=sum(np.min(D, axis=1))
    print "sum is", sum_1
    mean=float(sum_1)/float(X.shape[0])
    print "mean for k=",k," is", mean
    #print "pred labels", pred_labels
    #acc=find_accuracy(labels,pred_labels)
    #ACCURACY_train[k]=acc
    #pred=model.predict(X_test)
    #return pred

    

#list of k
k=[12,18,24,36,42]

for i in k:
    perform_k_means(i, df_rescaled, labels)#), X_test)

