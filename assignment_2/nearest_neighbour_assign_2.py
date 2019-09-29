#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:30:19 2018

@author:Ananya Banerjee
"""

import numpy as np
from sklearn import preprocessing
#Nearest neighbor Methods
K = [1 ,5, 11, 15, 21]

#Train dataset
#dataset contains the wntire data as lists is lists
#each list contains the value of a row
dataset_train=[] 


#Validation dataset
#dataset contains the wntire data as lists is lists
#each list contains the value of a row
dataset_val=[] 


#Test dataset
#dataset contains the wntire data as lists is lists
#each list contains the value of a row
dataset_test=[] 

#list of accuracy for train
ACCURACY_train=[]

#list of accuracy for validation
ACCURACY_val=[]

#list of accuracy for test
ACCURACY_test=[]

def MAIN_train():
    print "Training Set"
    #train file name
    file_name_train="/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_train.data"
    #open train file
    open_file(file_name_train, 'train')
    #now dataset_train contains the train data
    
      #forming vectors
    for k in K:
        y=[dataset_train[i][0] for i in range(len(dataset_train))]
        y = [x for x in y if x != '']
        y=np.array(y)
        X=[np.array(dataset_train[i]) for i in range(len(dataset_train))]
        del X[len(X)-1]
        X=np.array(X)
        #replacing 0 with -1
        #np.place(y, y==0, [-1])
        #print "y is",y
        knearest_neighbors(X, y, k)

def MAIN_validation():
    print "Validation Set"
    #train file name
    file_name_train="/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_train.data"
    #open val file
    open_file(file_name_train,'train')

    
    #validation file name
    file_name_val="/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_validation.data"
    #open val file
    open_file(file_name_val,'val')

    
    #forming vectors
    for k in K:
        y=[dataset_train[i][0] for i in range(len(dataset_train))]
        y = [x for x in y if x != '']
        y=np.array(y)
        X=[np.array(dataset_train[i]) for i in range(len(dataset_train))]
        del X[len(X)-1]
        X=np.array(X)
        #replacing 0 with -1
        #np.place(y, y==0, [-1])
        #print "y is",y
        neigh=knearest_neighbors_val(X, y, k)
        y_val=[dataset_val[i][0] for i in range(len(dataset_val))]
        y_val = [x for x in y_val if x != '']
        y_val=np.array(y_val)
        X_val=[np.array(dataset_val[i]) for i in range(len(dataset_val))]
        del X_val[len(X_val)-1]
        X_val=np.array(X_val)
        y_pred=neigh.predict(X_val)
        acc=calc_accuracy(y_pred, y_val)
        print "Accuracy is",acc
        ACCURACY_val.append(acc)
    
    print ACCURACY_val  
    print "Best value of K using validation set is", K[ACCURACY_val.index(np.max(ACCURACY_val))], "and the accuracy is",np.max(ACCURACY_val) 
    #test file name
    file_name_test="/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_test.data"
    #open test file
    open_file(file_name_test,'test')
    
    k=1
    y=[dataset_train[i][0] for i in range(len(dataset_train))]
    y = [x for x in y if x != '']
    y=np.array(y)
    X=[np.array(dataset_train[i]) for i in range(len(dataset_train))]
    del X[len(X)-1]
    X=np.array(X)
    knearest_neighbors(X, y, k)
   

#normalzie data
def normalize_data(X):
    X=preprocessing.normalize(X)
    return X


#function to open file and read the data
def open_file(file_name, file_type):
    
    file1 = open(file_name, 'r')
    
    # open the file for reading
    #filehandle = open(filename, 'r')  
    while True:  
        # read a single line
        line = file1.readline().replace('\n', '').replace('\r','')
        lines = line.split(',')
        #print "lines", type(lines), lines
        if lines!=['']:
            lines=[float(i) for i in lines]
        #data=lines.replace('\n', '')
        if file_type=='train':
           dataset_train.append(lines)
        
        elif file_type=='val':
           dataset_val.append(lines)
        
        else:
           dataset_test.append(lines) 
        # print len(lines), lines
        if not line:
            break
        #print(line)
    #print "hey",len(dataset_train), len(dataset_train[0])

def MAIN_test():
    print "Test Set"
    #test file name
    file_name_test="/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_test.data"
    #open test file
    open_file(file_name_test,'test')
    
    #forming vectors
    for k in K:
        y=[dataset_test[i][0] for i in range(len(dataset_test))]
        y = [x for x in y if x != '']
        y=np.array(y)
        X=[np.array(dataset_test[i]) for i in range(len(dataset_test))]
        del X[len(X)-1]
        X=np.array(X)
        #replacing 0 with -1
        #np.place(y, y==0, [-1])
        #print "y is",y
        knearest_neighbors(X, y, k)


#k-nearest neighbour 
def knearest_neighbors(X, y, k):
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y) 
    y_pred=neigh.predict(X)
    acc=calc_accuracy(y_pred, y)
    print "Accuracy is",acc
    ACCURACY_train.append(acc)
    #print neigh.predict([[1.1]])

#k-nearest neighbour 
def knearest_neighbors_val(X, y, k):
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y) 
    return neigh
    #print neigh.predict([[1.1]])
#function to calculate accuracy
def calc_accuracy(y_pred, y_true):
    correctly_labelled=0 
    #np.place(y_true, y_true==-1, [0])
    for i in range(len(y_pred)):
        #print "y_true",y_true[i]
        #print "y_pred", y_pred[i]
        if int(y_pred[i])==int(y_true[i]):
            correctly_labelled=correctly_labelled+1
        
    print "correctly_labelled", correctly_labelled
    accuracy= float(float(correctly_labelled)/float(len(y_pred)))*100
    return accuracy
 
#MAIN_train()
MAIN_validation()
#MAIN_test()