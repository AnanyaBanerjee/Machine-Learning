#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 12:27:31 2018

@author: Ananya Banerjee


Principal Component Analysis(PCA)

"""

import numpy as np
from numpy import linalg as lg
import pandas as pd
import math
from cvxopt import matrix, solvers  #this package helps solve quadratic optimization problems
from sklearn import svm, metrics


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

#dataset_val=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/assignment_4/sonar_valid.csv")

#function to find the similarity matrix 
def similarity_matrix(df):
    n,m=df.shape
    w=np.zeros((m,m))
    for i in range(m):
        #print "col is", df[i]
        mu=float(np.sum(df[:,i])/n)
        #print "mu is", mu
        for j in range(df.shape[1]):
            #print df[i].iloc[j]
            w[j][i]=df[j][i]-mu
           
    w=np.dot(w.T,w)   
    #print "shape of w", w.shape
    return w

#function to perform svd
def perform_svd(w):
    #u is col space: take columns--using  df[:k]
    #v is row space:take rows- using-df[a:b] chooses rows from a to b
    u, sig, v=lg.svd(w)
    return u, sig, v


#function to choose top k eigen vectors
def choose_k_eigen_vec(k, eigen_v):
    #shape of eigen vector is (60,60)
    return eigen_v[:k]



#function to choose top k eigen vectors
def choose_k_eigen_val(k, eigen_val):
    #shape of eigen vector is (60,60)
    return eigen_val[:k]**2  #squaring gives eigen values acc to numpy lib


#function to process data--applying pca
def apply_pca(df, k, c, labels):
    W=similarity_matrix(df)   
    #print "Similarity matrix is", W
    U, Sig, V=perform_svd(W)

    #chosen eigen vector
    chosen_eigen_vector=choose_k_eigen_vec(k, V)

    #chosen eigen val
    chosen_eigen_val=choose_k_eigen_val(k, Sig)
    #print "Chosen eigen val is", chosen_eigen_val

    #project data in k-dim space using V
    df_projected_V=project_space(k, df, V)

    #print "Shape of k=", k, " projected data space is", df_projected_V.shape
    #apply svm with slack

    return df_projected_V, V
    

#predict_test
def predict_test(w, b, X, C, alpha):
    y_pred=[]
    w=w.reshape(w.shape[0],1)
    #b=b.reshape(b.shape[0],1)
    #print X.shape, w.shape, b.shape
    y=np.dot(X,w)+b
    y=np.ravel(y)
    #print "y is",y
    #print y, len(y)
    for i in range(len(y)):
        #print y[i], " c is", C
        if int(np.sign(y[i]))==1:
            y_pred.append(1)
        else:
            y_pred.append(-1)
    return y_pred

#function to calculate accuracy
def calc_accuracy(y_pred, y_true):
    #print "in acc"
    correctly_labelled=0 
    #print "true",y_true
    #print "pred",y_pred
        
    #np.place(y_true, y_true==-1, [0])
    for i in range(len(y_pred)):
       if i==104:
           break;
       if int(y_pred[i])==int(y_true[i]):
            correctly_labelled=correctly_labelled+1
            #print "yay"
        
    print "correctly_labelled", correctly_labelled
    accuracy= float(float(correctly_labelled)/float(len(y_pred)))*100
    return accuracy

#function to apply kernel to X
def apply_kernel(kernel_name, X, sigma):
    #calculate K(x_i, x_j)
    if kernel_name=="linear kernel":
        n_samples=len(X)
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = linear_kernel(X[i], X[j])
    
    if kernel_name=="special kernel":
        n_samples=len(X)
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = special_kernel(X[i], X[j])
    
    if kernel_name=="polynomial kernel":
        n_samples=len(X)
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = polynomial_kernel(X[i], X[j])
    
    if kernel_name=="gaussian kernel":
        n_samples=len(X)
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = gaussian_kernel(X[i], X[j], sigma)
    
    return K

#kernel functions
    
#special kernel
def special_kernel(x_i, x_j,d):
    #(x_i.t *x_j)^d 
    return np.power(np.dot(x_i.T, x_j),d)

#linear kernel    
def linear_kernel(x_i, x_j):
    return np.dot(x_i, x_j)

#polynomial kernel
def polynomial_kernel(x_i, x_j, p=3):
    return (1 + np.dot(x_i,x_j)) ** p

#gaussian kernel
def gaussian_kernel(x_i, x_j, sigma):
    return np.exp(-np.linalg.norm(x_i-x_j)**2 / (2 * (sigma ** 2)))


#SVM function: to find w and b
def SVM(X, y, C, kernel_name):
    #print " y is", y.shape
    #print "C is", C
    #print "shape is", X.shape
    m=len(X)
    n=len(X[0])
    #y=y.reshape(-1,1)
    #print " y after is", y.shape
    ######
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
    ######
    K=apply_kernel(kernel_name, X, 0.0)
    #print " k is", K.shape
    H = np.outer(y, y)
    #print "H is", H.shape
    P = matrix(np.multiply(H,K))
    #P=matrix(vb1)
    #print "P is", P
    q = matrix(-np.ones((m, 1)))
    G = matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    #b=matrix(0.0)
    #running the solver:
    sol =solvers.qp(P, q, G, h, A, b)
    #extracting support vectors 
    # Lagrange multipliers
    a=np.array(sol['x'])
    # Support vectors have non zero lagrange multipliers
    #sv=np.where((1e-5<a) & (a<C*0.9))
    sv=np.where(a);
    #print "sv is", np.where()
    # Intercept
    b = 0
    #w
    w=np.zeros(X.shape[1])
    for i in range(m):
        #print "hey", a[i], y[i], X[i]
        w+=a[i]*y[i]*X[i]
       
    for i in range(len(sv)):
        b=y[sv[0]]-np.dot(X[sv[0]],w)
        #print "b is",b
    
   
    print("Learned Parameters - W:", w)
    #print "b is", b, len(b),b[0], type(b), 
    #print("Learned Parameters - B:", b[0][0])
    print("Learned Parameters - B:", b[0:6])
    
    return w, b[0:6], a
    #return w, b[0][0], a



def perform_SVM(X,y):
    #vaues of C
    C=[1,10,100,1000]
    #value of sigma
    sigma=[0.1, 1, 10, 100, 1000]
    
    for c in C:
        print "y is", len(y)
        #y=[dataset_train[i][0] for i in range(len(dataset_train))]
        #y = [x for x in y if x != '']
        #y=np.array(y)
        #X=[np.array(dataset_train[i]) for i in range(len(dataset_train))]
        #del X[len(X)-1]
        #X=[list(dataset_train[i]) for i in range(len(dataset_train))]
        #del X[len(X)-1]
        #print "letgh is",len(X), len(X[0])
        #for i in X:
        #    del i[0]
        #X=[np.array(i) for i in X]
    
        #X=np.array(X)
        #replacing 0 with -1
        #np.place(y, y==0, [-1])
        #print "y is",y
        w, b, alpha=SVM(X, y, c,'linear kernel')
        y_pred=predict_test(w, b, X, c, alpha)
        #print "y_pred is", y_pred
        acc=calc_accuracy(y_pred, y)
        #print "Accuracy is", acc
        #t=metrics.accuracy_score(y_pred,y)
        #print "t is", t
        #print "alpha is", alpha, len(alpha), type(alpha)
        ACCURACY_train.append(acc)
        
        #break;
    print ACCURACY_train



#function to test in sklearn's svm
def skl_svm(X, y, k, c, x_val,y_Val):
    clf= svm.SVC(gamma='auto', C=c)
    model=clf.fit(X,y)
    y_pred=model.predict(x_val)
    acc=metrics.accuracy_score(y_pred,y_Val)
    print "The accuracy for k=", k, "and c=", c," is ",acc*100
    return acc*100, model
    


#function to project data
def project_space(k, data, eigen_vec):
    #project data in k-dim space using V
    df_projected_V=np.dot(data,(eigen_vec[:k].T))
    return df_projected_V


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
       
#print labels
        
k=[6,5,4,3,2,1]
c=[1,10,100,1000]

#dictionary to hold {'k': [c, accuarcy]}
ACC_val=dict()

#dictionary to hold all ACC where x is iteration number and val is list of k and c and acc
GL_val=dict()
x=0

#list of model
MODEL=[]

print "Train->Val"
for i in k:
    for j in c:
        print "k=",i," c=",j
        projected_data, V=apply_pca(df, i, j, labels)  #apply pca, returns transformed space and eigen vecyor
        projected_val_data=project_space(i, df_val, V)  #transform validation set using the eigen vectors from training pca
        #projected_val_data=apply_pca(df_val, i, j, y_val)
        acc, model=skl_svm(projected_data, labels, i, j, projected_val_data,y_val) #apply svm and get accuracy tested on validation test
        ACC_val[i]=[j, acc]
        MODEL.append(model)
    GL_val[x]=ACC_val
    x=x+1  



#dictionary to hold {'k': [c, accuarcy]}
ACC_test=dict()

#dictionary to hold all ACC where x is iteration number and  val is list of k and c and acc
GL_test=dict()
x=0
itr=0
print "Train->Test"
for i in k:
    for j in c:
        projected_data, V=apply_pca(df, i, j, labels)  #apply pca, returns transformed space and eigen vecyor
        projected_test_data=project_space(i, df_test, V) #transform test set using the eigen vectors from training pca
        #projected_test_data=apply_pca(df_test, i, j, y_test)
        y_test_pred=MODEL[itr].predict(projected_test_data)
        itr+=1
        #acc=skl_svm(projected_data, labels, i, j, projected_test_data ,y_test)  #apply svm and get accuracy tested on test test
        acc=metrics.accuracy_score(y_test_pred,y_test)
        print "The accuracy for k=", i, "and c=", j," is ",acc*100
        ACC_test[i]=[j, acc*100]
    GL_test[x]=ACC_test
    x=x+1  
    


W=similarity_matrix(df)
print "Similarity matrix is", W
U, Sig, V=perform_svd(W)

#k=[1,2,3,4,5,6]
k=6

#chosen eigen vector
chosen_eigen_vector=choose_k_eigen_vec(k, V)

#chosen eigen val
chosen_eigen_val=choose_k_eigen_val(k, Sig)
print "Chosen eigen val is", chosen_eigen_val

#project data in k-dim space using V
df_projected_V=project_space(k, df, V)

print "Shape of k=", k, " projected data space is", df_projected_V.shape
#apply svm with slack

#perform_SVM(df_projected_V,labels)
c=10
#acc=skl_svm(df_projected_V,labels,k,y_val, dataset_val, c)
#y_pred_f=MODEL[21].predict(df_projected_V)
        
##acc=metrics.accuracy_score(y_pred_f,labels)
#print "The accuracy for k=", k, "and c=", c," is ",acc*100
    
#without feature selection
acc, mod=skl_svm(df, labels, 6, 10, df_val,y_val)
print "accuracy without feature selction for validation is", acc

acc, mod=skl_svm(df, labels, 6, 10, df_test,y_test)
print "accuracy without feature selction for test set is", acc


