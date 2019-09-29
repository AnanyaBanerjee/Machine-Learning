#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 23:53:27 2018

@author: Ananya Banerjee

Gaussian Mixture Model
(GMM)
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
import operator
import functools
import random

#Accuracy Train: key: k, val"accuracy"
ACCURACY_train=dict()

#dictionary containing y={1,2,..k} as key and its data likelihood as value
DATA_LIKELIHOOD=dict()

#dictionary containing q as values and y={1,2,,..k} as key
Q=dict()

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

#function to generate random values
def generate_random_val(size):
    u=[]
    for i in range(size):
        g=np.random.uniform(0,1)
        u.append(g)
        
    return u
       

#function to multiply element to alist
def mul(element, my_list):
    my_new_list = [i * element for i in my_list]
    return my_new_list

#function to divide element to alist
def div(element, my_list):
    my_new_list = [float(i)/float(element) for i in my_list]
    return my_new_list
 

#function to divide two lists
def divide(my_list_1, my_list_2):
    my_new_list = [x/y for x, y in zip(my_list_1, my_list_2)]
    return my_new_list
 
#function to subtract two lists
def sub(my_list_1, my_list_2):
    my_new_list = [x-y for x, y in zip(my_list_1, my_list_2)]
    return my_new_list
    


#function to calculate mean
def find_mean(data):
    #mu_final contains list of mean for each col
    mu_final=[]
    for i in range(data.shape[1]):
        mu=np.sum(data[:,i])
        #print mu
        mu_final.append(float(mu)/float(data.shape[0]))
        
    return mu_final

#function to calculate covariance matrix sigma
def find_cov_mat(data):
    #sig is covariance matrix of size (n,n)
    n,m=data.shape
    
    sig=np.zeros((m,m))
    
    mu_final=find_mean(data)
    
    for i in range(m):
        for j in range(m):
            sig[j][i]=df[j][i]-mu_final[i]
           
    sig=np.dot(sig.T,sig)   
    #print "shape of w", w.shape
    return sig, mu_final


#function to check if values in cov matrix are beyond a certain lower limit
def check_val(cov_mat):
    small=0.0000001
    #n=cov_mat.shape[0]
    n1=len(cov_mat)
    for i in range(n1):
        for j in range(n1):
            if cov_mat[i][j]<small:
                cov_mat[i][j]=small

    return cov_mat

#function to muiltply all elements of the list
def multiplyList(myList) : 
      
    # Multiply elements one by one 
    result = 1
    for x in myList: 
         result = result * x  
    return result  

#function to find data likelihood p(x/theta)
""" theta={mu_k, sig_k}  """
def find_data_likelihood(data, mu_k, sig_k):
    sig_k=check_val(sig_k)
    dete=lg.det(sig_k)
    d=data.shape[1] # d is dimension of data:==> no of features 
    n1=np.power(2*math.pi, d)
    n2=np.multiply(n1, dete)
    n3=1.0/float(np.sqrt(n2))
    #print "n3",n3
    #print "dete", dete
    dete_inv=lg.inv(sig_k) #inverse of sig
    #print "inverse is", dete_inv.shape
    #prob list of length n=340 i.e., total number of datapoints
    prob=[]
    for i in range(data.shape[0]):
        x=data[i]
        c=map(operator.sub, x, mu_k) 
        c=np.array(c)
        c=c.reshape(len(c),1)
        #print "c is",c.shape#,c
        c_t=np.transpose([c])
        c_t=c_t.reshape(1,len(c))
        #print "transose is", c_t.shape#, c_t
        #print "inv is", dete_inv.shape
        c0=np.matmul(c_t,dete_inv)
        c1=np.matmul(c0,c)
        #print "mul is",(np.matmul(c_t,dete_inv)).shape
        #print "c1", c1, c1.shape
        c2=-0.5*c1
        #print "c2 is", len(c2), c2
        fin=n3*np.exp(c2)
        #print "len of fin", len(fin), fin.shape
        #fin1=functools.reduce(operator.mul, fin, 1)
        fin1=multiplyList(fin)
        #print "fin1 is", fin1
        #fin2=multiplyList(fin1)
        #print "fin2 os", fin2
        prob.append(fin1)
    
    
    #final_prob=functools.reduce(operator.mul, prob, 1)
    #print "final prob is", prob, len(prob)
    
    return prob


""" Finding initial gaussain distribution and data likeliood of every data point to that cluster /distribution  """
def find_initial_distributions(k, df_rescaled):
    for i in range(k):
        """ Corresponding to each y={1,2,..k} we find intial mean, cov and hence data likelihood  """
        #random mean
        mean=generate_random_val(df_rescaled.shape[1])
        #random covariance matrix signa 
        sigma=np.eye(df_rescaled.shape[1],df_rescaled.shape[1])
        #calculate data likelihood
        data_lhood=find_data_likelihood(df_rescaled, mean, sigma)
        #add this to dictionary
        DATA_LIKELIHOOD[i]=data_lhood
      

""" Finding subsequent gaussain distribution and data likeliood of every data point to that cluster /distribution  """
def find_maxi_distributions(k, df_rescaled, mean, sigma):
    for i in range(k):
        """ Corresponding to each y={1,2,..k} we find intial mean, cov and hence data likelihood  """
        data_lhood=find_data_likelihood(df_rescaled, mean, sigma)
        #add this to dictionary
        DATA_LIKELIHOOD[i]=data_lhood
      
#function to sum2 lists
def sum_list(l1, l2):
    l3=[x+y for x, y in zip(l1,l2)]
    return l3



""" Finding updated mu and sigma and lambda in EM algo  """
def find_mu_sig_lambda(q, data, itr):
    sum1=[0 for i in range(data.shape[0])]
    vb=np.sum(q[itr])
    vb1=np.sum(q[:,itr]) #sum all q_i for one y
    #print "len q", len(q), len(data[0])
    #traverse every data point and find new mu, sig and lambda_y
    for j in range(data.shape[0]):
        mu=np.multiply(vb,data[j])
        sum1=sum_list(sum1, mu)
        #vb1+=
    
    ##calculating mean
    if vb1!=0:
       mean=[float(x)/float(vb1) for x in sum1] #list of means for every attr
    else:
       mean=[x for x in sum1]
       
    print "mean is", mean
    #calculating sigma
    sum_2=[0.0 for i in range(data.shape[0])]
    
    for j in range(data.shape[0]):
        s=np.array(sub(data[j], mean))
        s_t=np.array(np.transpose([s]))
        #print "s_t", s_t.shape, "s is", s.shape,"vb",vb
        s=s.reshape((len(s),1))
        s_t=s_t.reshape((1,len(s_t)))
        su1=np.matmul(s, s_t)
        #print "su1",su1.shape
        su2=np.multiply(su1, vb)
        #print "su2" ,su2.shape
        """
        su0=np.multiply(vb,s)
        su0=su0.reshape(len(su0),1)
        #su0=su0.reshape(1,len(su0))
        print "su0",su0.shape
        su=np.multiply(su0,s_t)
        print "su", su.shape
        """
        sum_2=sum_list(su2, sum_2)
    
    #print "sum_2 is", sum_2, len(sum_2), len(sum_2[0])
    #sigma=divide(sum_2,np.sum(q))
    if vb1!=0:
       sigma=np.divide(sum_2, vb1)
       #sigma=[float(x)/float(vb1) for x in sum_2] #list of means for every attr
    else:
       sigma=[x for x in sum_2]
    
    #calculation lambda
    sum_3=float(np.sum(q[:,itr]))/float(data.shape[0])
    lambda_y=sum_3

    return mean, sigma, lambda_y


#function to find lambda_y*datalikelood for every model or distribution or cluster
def calc_q_den(lambda_y,itr, data):
    sum1=[0 for i in range(data.shape[0])]
    #print "len of lambda", len(lambda_y)
    for i in range(len(lambda_y)):
        #print "lambda", lambda_y[i]
        v=mul(lambda_y[i], DATA_LIKELIHOOD[i])
        sum1+=v
       
    return sum1


#function to extract data point
def extract_data_lhood(j):
    l=[]
    for i in DATA_LIKELIHOOD.keys():
        v=DATA_LIKELIHOOD[i]
        l.append(v[j])
    
    #l gives the list of data likelood of each distribution of j'th data point
    return l

#function to create mat using data likelihood and lambda_y for each val
def create_mat(lambda_y, itr):
    """mat is list of len(k)  """
    l=extract_data_lhood(itr) #gives list of data likilood of one row corresponding to each distribution
    mat=[a*b for a,b in zip(l,lambda_y)]
    
    #print "mat is", mat, len(mat)
    return mat

#function to calculate q
def calculate_q_i(k, data, lambda_y):
    #another mat contains the matrix mat corresponding to each row and all distributions
    another_mat=[]
    #Q will be of shape (340,k)
    Q=np.zeros((data.shape[0],k))
    for j in range(k):
        for i in range(data.shape[0]):
            t=create_mat(lambda_y, i)
            another_mat.append(t)
            #print "t is ", t
            if np.sum(t)!=0.0:
               q_i=float(t[0])/float(np.sum(t))
            else: q_i=0.0
            Q[i][j]=q_i
        
    #print "Q is", len(Q)
    #print "shape of another_mat", len(another_mat), len(another_mat[0])
    
    return Q, another_mat


""" function to calculate log likelihood """
def log_likelihood(itr, k, Q, data, another_mat):
    d=DATA_LIKELIHOOD[itr]
    
    sum1=0.0
    
    for i in range(data.shape[0]):
        for y in range(k):
            q=Q[y]
            qd=divide(d, q)
            q1=np.multiply(q, qd)
            sum1+=q1
            
    print "value of log likelihood is", np.sum(sum1)
    return np.sum(sum1)
     
    
            
    
    


#function to perform em
def perform_expectation_max(k, data, max_iter):
    #creating initial distributions
    find_initial_distributions(k, data)
    #initial lambda y corresponding to every y: p(Y=y/theta)
    lambda_y=generate_random_val(k)
    
    #find q for every data point and for every val of y={1,2,,.,..k}
    #final_itr
    final_itr=0
    for n in range(max_iter):
        for i in range(k):
            """ Expectation Step  """
            #print "data", len(DATA_LIKELIHOOD[i][0])
            #Q gives (340,k) matrix
            Q, another_mat=calculate_q_i(k, data, lambda_y)
           
            """Maximization Step"""
            mean, sigma, lamb=find_mu_sig_lambda(Q,data, i)
            lambda_y[i]=lamb
            print "mean is", mean, "sigma is", sigma, len(sigma), "lamda changed", lamb
            find_maxi_distributions(k, data, mean, sigma)
        
        final_itr+=1          
        llhood=log_likelihood(itr, k, Q, data, another_mat)
        if llhood==0.0000001:
            break
            
            
    return mean, sigma, lambda_y, final_itr   
            
   


#cov_mat, mu_final=find_cov_mat(df_rescaled)  
#print cov_mat
#data_likelihood, q=find_data_likelihood(df_rescaled, mu_final, cov_mat)

#print data_likelihood

#dictionary to store k as key and mean and sigma and max_itr and lambda list as value list
FINAL=dict()

#k-values list
K=[12, 18, 24, 36, 42]

for k in K:   
    itr=10
    mean, sigma, lambda_y, final_itr=perform_expectation_max(k, df_rescaled, itr)
    FINAL[k]=[mean, sigma, lambda_y, final_itr]
   




        
    















