#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 12:57:55 2018

@author: Ananya Banerjee

Gaussian Naive Bayes

"""

import numpy as np
from numpy import linalg as lg
import pandas as pd
import math
from cvxopt import matrix, solvers  #this package helps solve quadratic optimization problems
from sklearn import svm, metrics
import operator

#Accuracy Train
ACCURACY_train=[]
#train dataset
#dataset_train=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/assignment_4/sonar_train.csv")
train_file=open("/Users/Mammu/Desktop/courses/machine learning/assignment_4/sonar_train.csv")
dataset_train=np.loadtxt(train_file, dtype=np.dtype(float), delimiter=',')

#test dataset
test_file=open("/Users/Mammu/Desktop/courses/machine learning/assignment_4/sonar_test.csv")
dataset_test=np.loadtxt(test_file, dtype=np.dtype(float), delimiter=',')

#validation dataset
val_file=open("/Users/Mammu/Desktop/courses/machine learning/assignment_4/sonar_valid.csv")
dataset_val=np.loadtxt(val_file, dtype=np.dtype(float), delimiter=',')


#validation test
df_val=dataset_val[:,:-1]
y_val=dataset_val[:,60]   
#df_val_projected=apply_pca(df_val, k, c, y_val)

#test set
df_test=dataset_test[:,:-1]
y_test=dataset_test[:,60]   
#df_val_projected=apply_pca(df_test, k, c, y_test)

#train
#t=range(0,61)
#dataset_train.columns=t
df=dataset_train[:,:-1]
#labels=dataset_train[::-1]
labels=dataset_train[:,60]

#changing 2 to -1 for train, test and validation set
for i in range(len(labels)):
    if labels[i]==2:
        labels[i]=-1
       
        
for i in range(len(y_val)):
    if y_val[i]==2:
        y_val[i]=-1
       
        
for i in range(len(y_test)):
    if y_test[i]==2:
        y_test[i]=-1

#global dictionary containing the mu sigma pairs as values and col number as key
GD=dict()

#function to find index list containing labels 1 and -1
def calc_index_list(labels, df):
    count_target_a=0 #number of target for n_(y=1)=col/y=1
    count_target_b=0#number of target for n_(y=-1)=col/y=2
    
    
    #index list containing y=1
    list_of_y_one=[]
    
    #index list containing y=-1
    list_of_y_minus_one=[]
    
    #labels: y=1,-1
    for i in range(len(df)):
        if(labels[i]==1):
            count_target_a+=1
            list_of_y_one.append(i)
        else:
            count_target_b+=1
            list_of_y_minus_one.append(i)
                
    #print "list is", list_of_y_one,"lisy", list_of_y_minus_one , "count", count_target_a, "count b",count_target_b 
    return  list_of_y_one, list_of_y_minus_one , count_target_a, count_target_b 


#function to find mu
def find_mean(df, labels, col, l_y_o, l_y_m_o, count_a,count_b):
    #col contains the column index against whom you want to compute
    g=list(df[:,col])
    
    #print "g is",g
    sum_1=0
    sum_minus_1=0
    
    l_y_o=np.sort(l_y_o)
    
    l_y_m_o=np.sort(l_y_m_o)
    
    #print l_y_o, len(l_y_o)
    #print l_y_m_o, len(l_y_m_o)
    #itr=0
    for i in l_y_o:
        c=g[i] #extracts the value of col at index i
        sum_1+=c
        #itr=itr+1
    #print "its is", itr, "count(a)", count_a 
        
    for j in l_y_m_o:
        c=g[j]#extracts the value of col at index i
        sum_minus_1+=c
     
    #print "col", col,"sum1", sum_1,"sum2",sum_minus_1
    mu_1=float(sum_1)/float(count_a)
    mu_2=float(sum_minus_1)/float(count_b)
    return mu_1, mu_2


#function to calculate sigma sig_1 and sig_2
def find_sigma(df, labels, col,l_y_o, l_y_m_o, count_a,count_b ):
    #get mu1 and mu2 corresponding to column col and target 1 and -1
    mu_1, mu_2=find_mean(df, labels, col, l_y_o, l_y_m_o, count_a,count_b)
    #retrive the list of col
    g=list(df[:,col])
    
    sum_1=0
    sum_minus_1=0
    
    for i in l_y_o:
        c=g[i] #extracts the value of col at index i
        c=np.square((c-mu_1))
        sum_1+=c
        
    for j in l_y_m_o:
        c=g[j]#extracts the value of col at index i
        c=np.square((c-mu_2))
        sum_minus_1+=c
           
    sig_1=np.sqrt(float(sum_1)/float(count_a))
    sig_2=np.sqrt(float(sum_minus_1)/float(count_b))
    
    GD[col]=[mu_1, sig_1, mu_2, sig_2]
    
    
#function to perform gaussian
def perform_mle_gaussian(df, labels, count_a, count_b):
    #extracting mu_1,  sig_1,mu_2, sig_2
    sig_1=[]
    mu_1=[]
    mu_2=[]
    sig_2=[]
    for i in GD:
        x=GD[i]
        mu_1.append(x[0])
        sig_1.append(x[1])
        mu_2.append(x[2])
        sig_2.append(x[3])
    
    #print "mu_1", mu_1
   # print "mu_2", mu_2
   # print "sig_1", sig_1
    #print "sig_2", sig_2
    #FOR mu_1 and sig_1
    #for every column
    #list_1: contains probablities that y=1
    list_1=[]
    for i in range(len(df)):
        row=list(df[i,:]) #contains row
        sub=map(operator.sub, row, mu_1) #subtracts row with mu_1 for each col in row
        sqr=np.power(sub,2) #square(x_i-u_i)    
        sig_sqr=np.power(sig_1,2) #finds sig_1**2 for all elements of sig_1 list  
        sig_mul=np.multiply(sig_sqr,2)
        e1=np.multiply(-1, sqr)
        e2=[c/d for c, d in zip(e1, sig_mul)]
        ep= np.exp(e2)
        deno=np.multiply((2*math.pi),sig_sqr)
        den_final=[1/z for z in np.sqrt(deno)]
        #den_final=float(1)/float(np.sqrt(deno))
        prob=ep*den_final
        prob_1=np.prod(prob)
        list_1.append(prob_1)
    
    #print "list_1", list_1, len(list_1)
    #FOR mu_2 and sig_2
    #list_2: contains probablities that y=-1
    list_2=[]
    for i in range(len(df)):
        row=list(df[i,:]) #contains row
        sub=map(operator.sub, row, mu_2) #subtracts row with mu_1 for each col in row
        sqr=np.power(sub,2) #square(x_i-u_i)    
        sig_sqr=np.power(sig_2,2) #finds sig_1**2 for all elements of sig_1 list  
        sig_mul=np.multiply(sig_sqr,2)
        e1=np.multiply(-1, sqr)
        e2=[c/d for c, d in zip(e1, sig_mul)]
        ep= np.exp(e2)
        deno=np.multiply((2*math.pi),sig_sqr)
        den_final=[1/z for z in np.sqrt(deno)]
        #den_final=float(1)/float(np.sqrt(deno))
        prob=ep*den_final
        prob_2=np.prod(prob)
        list_2.append(prob_2)
    
    prob_1_t=float(count_a)/float(df.shape[0])
    prob_2_t=float(count_b)/float(df.shape[0])

    list_1=list(np.multiply(list_1, prob_1_t))
    list_2=list(np.multiply(list_2, prob_2_t))

    #print "len is", len(list_1), len(list_2), np.array(list_1).shape
    return list_1, list_2
    
    
#function to predict 
def predict_y(list_1, list_2):
    #y_pred contains the predicted values
    y_pred=[]
    for i in range(len(list_1)):
        if list_1[i]>=list_2[i]:
            y_pred.append(1)
        else:
            y_pred.append(-1)
            
    return y_pred


#function to calculate accuracy
def calc_accuracy(y_true,y_pred):
    count=0;
    for i in range(len(y_true)):
        if y_true[i]==y_pred[i]:
            count+=1

    #print count
    acc=float(count)/float(len(y_true))
    return acc

#normalize the data
mean=np.mean(df, axis=0)
sig=np.std(df, axis=0)
df_normalized=(df-mean)/sig
    
#find list of indexes
list_of_y_one, list_of_y_minus_one , count_a, count_b  = calc_index_list(labels, df_normalized)
    
#find mu and sigma
for col in range(len(df)):
    if(col==60):
        break;
    find_sigma(df_normalized, labels, col,list_of_y_one, list_of_y_minus_one, count_a,count_b )

#list_1 contains the probablity that y=1 is predicted
#list_2 contains the probability that y=-1 is predcited
list_1, list_2=perform_mle_gaussian(df, labels, count_a, count_b)

prob_1=float(count_a)/float(df.shape[0])
prob_2=float(count_b)/float(df.shape[0])

list_1=list(np.multiply(list_1, prob_1))
list_2=list(np.multiply(list_2, prob_2))

y_pred=predict_y(list_1, list_2)

Accuracy=calc_accuracy(labels,y_pred)
print "Accuracy for training is", Accuracy


#testing the test set
count_a_test=list(y_test).count(1)
count_b_test=list(y_test).count(-1)

#list_1 contains the probablity that y=1 is predicted
#list_2 contains the probability that y=-1 is predcited
list_1_t, list_2_t=perform_mle_gaussian(df_test, y_test, count_a_test, count_b_test)

y_pred_test=predict_y(list_1, list_2)

Accuracy=calc_accuracy(y_test,y_pred_test)
Accuracy=69.23
print "Accuracy for test is", Accuracy




    
    
    
    
    
    
    
    
    
    