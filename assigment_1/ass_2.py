#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 23:38:19 2018

@author: Mammu
"""

#assignent 1: ques 4 svm
import pandas as pd
import numpy as np
import math
#from sympy import symbols, diff
from cvxopt import matrix, solvers  #this package helps solve quadratic optimization problems
from sklearn import svm



df=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/mystery.csv")


#function to calculate margin
def calculate_margin(w):
    norm_W = 0.
    for i in range(len(w)):
        norm_W += w[i] ** 2

    return 1 / math.sqrt(norm_W)

#function to form vectors
def form_vectors(X):
   arr_x=[]
   for i in range(X.shape[0]):
       #arr_x.append([X.iloc[i]["x1"],X.iloc[i]["x2"],X.iloc[i]["x3"],X.iloc[i]["x4"],X.iloc[i]["y"]])
       arr_x.append(list(X.iloc[i]))
       #arr_y.append([X.iloc[i]["y"]])
   return arr_x
       

#this function takes dataset and degree as input
#this transforms space into higher degree polynomial
def transform_space(X, degree):
    #f_v is the feature vector transformed space
    if degree==1:
       X = X.assign(x1= X.x1 **degree) #df1.x1.mul(df1.x1))
       X = X.assign(x2= X.x2 **degree) #df1.x1.mul(df1.x1))
       X = X.assign(x3= X.x3 **degree) #df1.x1.mul(df1.x1))
       X = X.assign(x4= X.x4 **degree) #df1.x1.mul(df1.x1))
    else:
       X=transform_space(X, degree-1)
       X["x1"+str(degree)]=X["x1"]**degree
       X["x2"+str(degree)]=X["x2"]**degree
       X["x3"+str(degree)]=X["x3"]**degree
       X["x4"+str(degree)]=X["x4"]**degree
    
    return X    #X is the feature space dataset





#SVM function: to find w and b
def SVM(X, y):
    print "X", X.shape
    m,n=X.shape
    y=y.reshape(-1,1)*1
    y=y.astype('d')
    ####
    #Dual problem:
    #min 1/2 * alpha.T * H * aplha - 1.T * aplha
    #s.t. -alpha_i <=0
    #s.t. y.T * alpha =0
    #P: H a matrix of mxm
    #where H= y*X * y*X.T
    #q: -1 vector of size mX1
    #G: -diag[1] a diag matric pf -1's os size mXm
    #h: a vector of zeroes of size mX1
    #A: y..the label vector of size mx1
    #b: scalar 0
    ####
    K = np.outer(y, y)
    H=np.dot(X, X.T)
    P = matrix(np.multiply(K,H))
    q = matrix(-np.ones((m, 1)))
    G = matrix(-np.eye(m))
    h = matrix(np.zeros(m))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    
    #running the solver:
    sol =solvers.qp(P, q, G, h, A, b)
    #alphas = np.array(sol['x'])
    
    #extracting support vectors 
    #w parameter in vectorized form
    
    # Lagrange multipliers
    a = np.ravel(sol['x'])
    # Support vectors have non zero lagrange multipliers
    sv = a > 1e-5
    ind = np.arange(len(a))[sv]
    # Get the corresponding lagr. multipliers
    a = a[sv]
    print "a is", a
    # Get the samples that will act as support vectors
    sv_ = x[sv]
    #print "support vectors", sv_, len(sv_)
    # Get the corresponding labels
    sv_y = y[sv]
    
    # Intercept
    b = 0
    for n in range(len(a)):
        b += sv_y[n]
        b -= np.sum(a * sv_y * K[ind[n], sv])
    b /= len(a)
    # Weight vector
    w = np.zeros(X.shape[1])
    for n in range(len(a)):
        w += a[n] * sv_y[n] * sv_[n]
    
    print("Learned Parameters - W:", w)
    print len(w)
    print("Learned Parameters - B:", b)
    
    return w, b
#Feature is a list of transfored space
#Feature=["x1", "x2", "x3", "x4", "y"] where each of these are transformed spaces
#Feature=transform_space(df, 2)
#flag decides if the feature space that we chose is good enough or not
flag=False
degree=6

while(flag!=True):
    Feature=transform_space(df, degree)
    y=np.array(Feature["y"])
    Feature=Feature.drop('y',1)
    x=Feature
    x=np.array(form_vectors(x)) #now x is a array containing [1,2,3,4] values
    
    w, b=SVM(x, y)
    margin=calculate_margin(w)
    print "Margin is", margin
    flag=True
    degree=degree-1;
    
    
   

#print "w is", w
#print "b is", b

#margin=calculate_margin(w) 
#print "the margin is", margin


"""
#using standard svm of sklearn
clf = svm.SVC()
clf.fit(X, Y)
#print clf.predict([[2., 2.]])
s_veec=clf.support_vectors_
#print s_veec
#print clf.score(X, y)

#print clf


def fit(x, y): 
    NUM = x.shape[0]
    DIM = x.shape[1]
    # we'll solve the dual
    # obtain the kernel
    K=np.zeros((NUM, NUM))
    #for i in range(NUM):
    #  for j in range(NUM):
    #      K[i,j] = np.dot(x[i], x[j])
    #K = y[:, None] * x
    K = np.dot(x, x.T)
    P = matrix(np.outer(y,y)*K)
    q = matrix(np.ones(NUM)*-1)
    #G = matrix(-np.eye(NUM))
    G=matrix(np.diag(np.ones(NUM)*-1))
    h = matrix(np.zeros(NUM))
    A = matrix(y, (1, NUM),'d')
    b = matrix(0.0)
    #solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    #print "sol is", sol
    #after sol    
    # Lagrange multipliers
    a = np.ravel(sol['x'])
    # Support vectors have non zero lagrange multipliers
    sv = a > 1e-5
    ind = np.arange(len(a))[sv]
    # Get the corresponding lagr. multipliers
    a = a[sv]
    # Get the samples that will act as support vectors
    sv_ = x[sv]
    print "support vectors", sv_, len(sv_)
    # Get the corresponding labels
    sv_y = y[sv]
    
    # Intercept
    b = 0
    for n in range(len(a)):
        b += sv_y[n]
        b -= np.sum(a * sv_y * K[ind[n], sv])
    b /= len(a)
    # Weight vector
    w = np.zeros(DIM)
    for n in range(len(a)):
        w += a[n] * sv_y[n] * sv_[n]
    return w, b

#x=df[['x1','x2','x3','x4']].values
x=df.iloc[:,0:4].values
#y=list(df["y"])
y=df.iloc[:,4].values
w, b=fit(x,y)

margin=1/np.sqrt(np.sum(w**2))
d=np.sqrt(np.sum(w**2))
print margin,d
"""