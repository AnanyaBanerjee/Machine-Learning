#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 20:09:19 2018

@author: Ananya Banerjee
"""



#assignent 1: ques 4 svm
import pandas as pd
import numpy as np
import math
#from sympy import symbols, diff
from cvxopt import matrix, solvers  #this package helps solve quadratic optimization problems
from sklearn import svm, metrics

solvers.options['show_progress']=False
"""
RUN MAIN_1() FOR PROBLEM 1: 1.(A) AND 1.(B)
RUN MAIN_2() FOR PROBLEM 1: 1.(C) 
RUN MAIN_3() FOR  PROBLEM 1: 1.(D)
RUN MAIN_4() FOR PROBLEM 1: 2.(A) AND 2.(B)
RUN MAIN_5() FOR PROBLEM 1: 2.(C) 
RUN_MAIN_6() FOR PROBLEM 1: 2.(D)
"""

#list containing all margins
Margin=[]

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

#train data
df_train=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_train.csv")
#validation data
df_validation=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_validation.csv")
#test data
df_test=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_test.csv")

print df_train.shape
print df_validation.shape
print df_test.shape  

def MAIN_1():
    
    #train file name
    file_name_train="/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_train.data"
    #open train file
    open_file(file_name_train, 'train')
    #now dataset_train contains the train data
    
    """
    #train data
    df_train=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_train.csv")
    #validation data
    df_validation=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_validation.csv")
    #test data
    df_test=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_test.csv")

    print df_train.shape
    print df_validation.shape
    print df_test.shape    
    """
    #vaues of C
    C=[1,10,100,1000,10000,100000,1000000,10000000,100000000]

    #value of sigma
    sigma=[0.1, 1, 10, 100, 1000]
    
    for c in C:
        y=[dataset_train[i][0] for i in range(len(dataset_train))]
        y = [x for x in y if x != '']
        y=np.array(y)
        #X=[np.array(dataset_train[i]) for i in range(len(dataset_train))]
        #del X[len(X)-1]
        X=[list(dataset_train[i]) for i in range(len(dataset_train))]
        del X[len(X)-1]
        #print "letgh is",len(X), len(X[0])
        for i in X:
            del i[0]
        X=[np.array(i) for i in X]
    
        X=np.array(X)
        #replacing 0 with -1
        np.place(y, y==0, [-1])
        #print "y is",y
        w, b, alpha=SVM(X, y, c,'linear kernel')
        margin=calculate_margin(w)
        print "margin is", margin, "for C", c
        Margin.append(margin)
        #w=w.reshape(23, 1)
        #print w, type(w), len(w), w.shape
        #y_pred=predict_1(w, b, X, c, alpha,y)
        #y_pred=predict(w, b, X, c, alpha)
        y_pred=predict_test(w, b, X, c, alpha)
        #print "y_pred is", y_pred
        acc=calc_accuracy(y_pred, y)
        print "Accuracy is", acc
        #t=metrics.accuracy_score(y_pred,y)
        #print "t is", t
        #print "alpha is", alpha, len(alpha), type(alpha)
        ACCURACY_train.append(acc)
        y_pred=0
        acc=0
        #break;
    print ACCURACY_train

def MAIN_2():
   
    #train file name
    file_name_train="/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_train.data"
    #open train file
    open_file(file_name_train, 'train')
    #now dataset_train contains the train data
    
    #validation file name
    file_name_val="/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_validation.data"
    #open val file
    open_file(file_name_val,'val')

    """
    #train data
    df_train=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_train.csv")
    #validation data
    df_validation=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_validation.csv")
    #test data
    df_test=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_test.csv")

    print df_train.shape
    print df_validation.shape
    print df_test.shape    
    """
    #vaues of C
    C=[1,10,100,1000,10000,100000,1000000,10000000,100000000]

    #value of sigma
    sigma=[0.1, 1, 10, 100, 1000]
    
    for c in C:
        y=[dataset_train[i][0] for i in range(len(dataset_train))]
        y = [x for x in y if x != '']
        y=np.array(y)
        #X=[np.array(dataset_train[i]) for i in range(len(dataset_train))]
        #del X[len(X)-1]
        X=[list(dataset_train[i]) for i in range(len(dataset_train))]
        del X[len(X)-1]
        #print "letgh is",len(X), len(X[0])
        for i in X:
            del i[0]
        X=[np.array(i) for i in X]
    
        X=np.array(X)
        #replacing 0 with -1
        np.place(y, y==0, [-1])
        #print "y is",y
        w, b, alpha=SVM(X, y, c,'linear kernel')
        #margin=calculate_margin(w)
        #print "margin is", margin, "for C", c
        #Margin.append(margin)
        
        y_val=[dataset_val[i][0] for i in range(len(dataset_val))]
        y_val = [x for x in y_val if x != '']
        y_val=np.array(y_val)
        X_val=[list(dataset_val[i]) for i in range(len(dataset_val))]
        del X_val[len(X_val)-1]
        #X_val=np.array(X_val)
        #replacing 0 with -1
        np.place(y_val, y_val==0, [-1])
        for i in X_val:
            del i[0]
        X_val=[np.array(i) for i in X_val]
        X_val=np.array(X_val)
        #print "y is",y
        #w=w.reshape(23, 1)
        #print w, type(w), len(w), w.shape
        #y_pred=predict_1(w, b, X, c, alpha,y)
        y_pred=predict_test(w, b, X_val, c, alpha)
        #print "y_pred is", y_pred
        acc=calc_accuracy(y_pred, y_val)
        #print "Accuracy is", acc
        #print "alpha is", alpha, len(alpha), type(alpha)
        ACCURACY_val.append(acc)
        #break;
    
    print ACCURACY_val  
    print "Best value of C using validation set is", C[ACCURACY_val.index(np.max(ACCURACY_val))], "and the accuracy is",np.max(ACCURACY_val) 

def MAIN_3():
    #train file name
    file_name_train="/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_train.data"
    #open train file
    open_file(file_name_train, 'train')
    #now dataset_train contains the train data
    
    
    #vaues of C
    #C=[1,10,100,1000,10000,100000,1000000,10000000,100000000]

    #value of sigma
    sigma=[0.1, 1, 10, 100, 1000]
    #c=100
    c=100
    #training the data
    y=[dataset_train[i][0] for i in range(len(dataset_train))]
    y = [x for x in y if x != '']
    y=np.array(y)
    #X=[np.array(dataset_train[i]) for i in range(len(dataset_train))]
    X=[list(dataset_train[i]) for i in range(len(dataset_train))]
    del X[len(X)-1]
    #print "letnh is",len(X), len(X[0])
    for i in X:
        del i[0]  #deleting y from X
    X=[np.array(i) for i in X]
    X=np.array(X)
    np.place(y, y==0, [-1])
        
    w, b, alpha=SVM(X, y, c,'linear kernel')
    #margin=calculate_margin(w)
    #print "margin on training set is", margin, "for C", c
    #Margin.append(margin)
    
    
    #print "middie"
    #validation file name
    file_name_test="/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_test.data"
    #open val file
    open_file(file_name_test,'test')
    #forming the vectors
    #print "length of test", len(dataset_test)
    y_test=[dataset_test[i][0] for i in range(len(dataset_test))]
    #print len(y_test), "leng is"
    y_test = [x for x in y_test if x != '']
    y_test=np.array(y_test)
    #print len(y_test), "leng is"
    #X_test=[np.array(dataset_test[i]) for i in range(len(dataset_test))]
    X_test=[dataset_test[i] for i in range(len(dataset_test))]
    del X_test[len(X_test)-1]
    for i in X_test:
            del i[0]
    X_test=[np.array(i) for i in X_test]
    
    X_test=np.array(X_test)
    #print " X_test is", X_test[0], len(X_test[0])
    #predicting the labels using the trained dataset
    #print y_test.shape, w.shape, b.shape, X_test.shape 
    #X_test=list(X_test)
    #del X_test[118] 
    X_test=np.array(X_test)
    
    #print y_test.shape, w.shape, b.shape, X_test.shape
    y_pred=predict_test(w, b, X_test, c, alpha)
    #print "y_pred is", y_pred
    acc=calc_accuracy(y_pred, y_test)
    print "Accuracy on the test data is", acc
    #print "alpha is", alpha, len(alpha), type(alpha)
    ACCURACY_test.append(acc)
    
    print  ACCURACY_test
    #print "Best value of C using validation set is", C[ACCURACY_val.index(np.max(ACCURACY_val))]

def MAIN_4():
    
    #train file name
    file_name_train="/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_train.data"
    #open train file
    open_file(file_name_train, 'train')
    #now dataset_train contains the train data
    
    """
    #train data
    df_train=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_train.csv")
    #validation data
    df_validation=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_validation.csv")
    #test data
    df_test=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_test.csv")

    print df_train.shape
    print df_validation.shape
    print df_test.shape    
    """
    #vaues of C
    C=[1,10,100,1000,10000,100000,1000000,10000000,100000000]

    #value of sigma
    sigma=[0.1, 1, 10, 100, 1000]
    
    #dictionary of list where c is keys and sigma and its corresponding accuracies are values
    Dict=dict()
    
    for c in C:
        y=[dataset_train[i][0] for i in range(len(dataset_train))]
        y = [x for x in y if x != '']
        y=np.array(y)
        #X=[np.array(dataset_train[i]) for i in range(len(dataset_train))]
        #del X[len(X)-1]
        X=[list(dataset_train[i]) for i in range(len(dataset_train))]
        del X[len(X)-1]
        #print "letgh is",len(X), len(X[0])
        for i in X:
            del i[0]
        X=[np.array(i) for i in X]
    
        X=np.array(X)
        #replacing 0 with -1
        np.place(y, y==0, [-1])
        #print "y is",y
        list_of_s=[]
        for sig in sigma:
            #dictionary containing sig and corresponding accuarcies
            s=dict()
            w, b, alpha=SVM_Gaussian(X, y, c,'gaussian kernel', sig)
            margin=calculate_margin(w)
            print "margin is", margin, "for C", c
            Margin.append(margin)
            #w=w.reshape(23, 1)
            #print w, type(w), len(w), w.shape
            #y_pred=predict_1(w, b, X, c, alpha,y)
            #y_pred=predict(w, b, X, c, alpha)
            y_pred=predict_test(w, b, X, c, alpha)
            #print "y_pred is", y_pred
            acc=calc_accuracy(y_pred, y)
            #print "Accuracy is", acc
            s[sig]=acc
            #print " s is",s
            #t=metrics.accuracy_score(y_pred,y)
            #print "t is", t
            #print "alpha is", alpha, len(alpha), type(alpha)
            ACCURACY_train.append(acc)
            y_pred=0
            acc=0
            list_of_s.append(s)
        Dict[c] =list_of_s    
    print "Dictionary contains C as keys and its values are lists of dictionary eachhaving sigma value and its corresponding accuracies as values"
    print Dict
        



def MAIN_5():
   
    #train file name
    file_name_train="/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_train.data"
    #open train file
    open_file(file_name_train, 'train')
    #now dataset_train contains the train data
    
    #validation file name
    file_name_val="/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_validation.data"
    #open val file
    open_file(file_name_val,'val')

    """
    #train data
    df_train=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_train.csv")
    #validation data
    df_validation=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_validation.csv")
    #test data
    df_test=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_test.csv")

    print df_train.shape
    print df_validation.shape
    print df_test.shape    
    """
    #vaues of C
    C=[1,10,100,1000,10000,100000,1000000,10000000,100000000]

    #value of sigma
    sigma=[0.1, 1, 10, 100, 1000]
    
    #dictionary of list where c is keys and sigma and its corresponding accuracies are values
    Dict=dict()
    
    for c in C:
        y=[dataset_train[i][0] for i in range(len(dataset_train))]
        y = [x for x in y if x != '']
        y=np.array(y)
        #X=[np.array(dataset_train[i]) for i in range(len(dataset_train))]
        #del X[len(X)-1]
        X=[list(dataset_train[i]) for i in range(len(dataset_train))]
        del X[len(X)-1]
        #print "letgh is",len(X), len(X[0])
        for i in X:
            del i[0]
        X=[np.array(i) for i in X]
    
        X=np.array(X)
        #replacing 0 with -1
        np.place(y, y==0, [-1])
        #print "y is",y
        list_of_s=[]
        for sig in sigma:
            #dictionary containing sig and corresponding accuarcies
            s=dict()
            w, b, alpha=SVM_Gaussian(X, y, c,'gaussian kernel', sig)
          
            y_val=[dataset_val[i][0] for i in range(len(dataset_val))]
            y_val = [x for x in y_val if x != '']
            y_val=np.array(y_val)
            X_val=[list(dataset_val[i]) for i in range(len(dataset_val))]
            del X_val[len(X_val)-1]
            #X_val=np.array(X_val)
            #replacing 0 with -1
            np.place(y_val, y_val==0, [-1])
            for i in X_val:
                del i[0]
            X_val=[np.array(i) for i in X_val]
            X_val=np.array(X_val)
            #print "y is",y
            #w=w.reshape(23, 1)
            #print w, type(w), len(w), w.shape
            #y_pred=predict_1(w, b, X, c, alpha,y)
            y_pred=predict_test(w, b, X_val, c, alpha)
            #print "y_pred is", y_pred
            acc=calc_accuracy(y_pred, y_val)
        
            s[sig]=acc
            #print " s is",s
            #t=metrics.accuracy_score(y_pred,y)
            #print "t is", t
            #print "alpha is", alpha, len(alpha), type(alpha)
            ACCURACY_val.append(acc)
            y_pred=0
            acc=0
            list_of_s.append(s)
        Dict[c] =list_of_s    
    print "Dictionary contains C as keys and its values are lists of dictionary eachhaving sigma value and its corresponding accuracies as values"
    print Dict
        
    print "Best value of C using validation set is", C[ACCURACY_val.index(np.max(ACCURACY_val))], "and the accuracy is",np.max(ACCURACY_val) 
    print "for sigma", 10

def MAIN_6():
    
    #train file name
    file_name_train="/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_train.data"
    #open train file
    open_file(file_name_train, 'train')
    #now dataset_train contains the train data
    
    #test file name
    file_name_test="/Users/Mammu/Desktop/courses/machine learning/assignment_2/park_test.data"
    #open train file
    open_file(file_name_train, 'test')
    #now dataset_train contains the train data
    
    
    #vaues of C
    #C=[1,10,100,1000,10000,100000,1000000,10000000,100000000]

    #value of sigma
    #sigma=[0.1, 1, 10, 100, 1000]
    #c=100
    c=10
    sigma=10
    #training the data
    y=[dataset_train[i][0] for i in range(len(dataset_train))]
    y = [x for x in y if x != '']
    y=np.array(y)
    #X=[np.array(dataset_train[i]) for i in range(len(dataset_train))]
    X=[list(dataset_train[i]) for i in range(len(dataset_train))]
    del X[len(X)-1]
    #print "letnh is",len(X), len(X[0])
    for i in X:
        del i[0]  #deleting y from X
    X=[np.array(i) for i in X]
    X=np.array(X)
    np.place(y, y==0, [-1])
        
    w, b, alpha=SVM_Gaussian(X, y, c,'gaussian kernel',sigma)
    #margin=calculate_margin(w)
    #print "margin on training set is", margin, "for C", c
    #Margin.append(margin)
    
    #forming the vectors
    #print "length of test", len(dataset_test)
    y_test=[dataset_test[i][0] for i in range(len(dataset_test))]
    #print len(y_test), "leng is"
    y_test = [x for x in y_test if x != '']
    y_test=np.array(y_test)
    #print len(y_test), "leng is"
    #X_test=[np.array(dataset_test[i]) for i in range(len(dataset_test))]
    X_test=[dataset_test[i] for i in range(len(dataset_test))]
    del X_test[len(X_test)-1]
    for i in X_test:
            del i[0]
    X_test=[np.array(i) for i in X_test]
    
    X_test=np.array(X_test)
    #print " X_test is", X_test[0], len(X_test[0])
    #predicting the labels using the trained dataset
    #print y_test.shape, w.shape, b.shape, X_test.shape 
    #X_test=list(X_test)
    #del X_test[118] 
    X_test=np.array(X_test)
    
    #print y_test.shape, w.shape, b.shape, X_test.shape
    y_pred=predict_test(w, b, X_test, c, alpha)
    #print "y_pred is", y_pred
    acc=calc_accuracy(y_pred, y_test)
    print "Accuracy on the test data is", acc
    #print "alpha is", alpha, len(alpha), type(alpha)
    ACCURACY_test.append(acc)
    
    #print "Best value of C using validation set is", C[ACCURACY_val.index(np.max(ACCURACY_val))]


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
        
        elif file_type=='test':
           dataset_test.append(lines) 
        # print len(lines), lines
        if not line:
            break
        #print(line)
    #print len(dataset_test), "len"
    #print "hey",len(dataset_train), len(dataset_train[0])



#function to calculate margin
def calculate_margin(w):
    norm_W = 0.
    for i in range(len(w)):
        norm_W += w[i] ** 2
    if norm_W==0:
       return 0
    return 1 / math.sqrt(norm_W)


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


#function to form vectors
def form_vectors(X):
   arr_x=[]
   for i in range(len(X)):
       arr_x.append(X[i])
   
   return arr_x
       
#gaussian kernel with svm
#SVM function: to find w and b
def SVM_Gaussian(X, y, C, kernel_name, sigma):
    print "C is", C
    m=len(X)
    n=len(X[0])
    y=y.reshape(-1,1)
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
    K=apply_kernel(kernel_name, X, sigma)
    H = np.outer(y, y)
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
    #alphas = np.array(sol['x'])
    
    #extracting support vectors 
    # Lagrange multipliers
    #a = np.ravel(sol['x'])
    a=np.array(sol['x'])
    #print " a is of shape", a.shape
    #print "a is", a, len(a)
    # Support vectors have non zero lagrange multipliers
    #sv=a < C # 0<aplha<=c
    #sv = sv > 1e-5
    #sv is indexes of support vectors
    sv=np.where((1e-5<a) & (a<C*0.9))
    #print "sv is", sv
    #ind = np.arange(len(a))[sv]
    #print "index is", ind
    # Get the corresponding lagr. multipliers
    #a = a[sv]
    # Get the samples that will act as support vectors
    #sv_ = X[sv]
    #print sv_
    #print "support vectors", sv_, len(sv_)
    # Get the corresponding labels
    #sv_y = y[sv]
    #print sv_y
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
    
    """
    for n in range(len(a)):
        b += sv_y[n]
        b -= np.sum(a * sv_y * K[ind[n], sv])
    
    b=b/a
    # Weight vector
    w = np.zeros(X.shape[1])
    for n in range(len(a)):
        w += a[n] * sv_y[n] * sv_[n]
    """
    print("Learned Parameters - W:", w)
    #print len(w)
    print("Learned Parameters - B:", b[0][0])
    
    return w, b[0][0], a


#SVM function: to find w and b
def SVM(X, y, C, kernel_name):
    print "C is", C
    m=len(X)
    n=len(X[0])
    y=y.reshape(-1,1)
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
    H = np.outer(y, y)
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
    #alphas = np.array(sol['x'])
    
    #extracting support vectors 
    # Lagrange multipliers
    #a = np.ravel(sol['x'])
    a=np.array(sol['x'])
    #print " a is of shape", a.shape
    #print "a is", a, len(a)
    # Support vectors have non zero lagrange multipliers
    #sv=a < C # 0<aplha<=c
    #sv = sv > 1e-5
    #sv is indexes of support vectors
    sv=np.where((1e-5<a) & (a<C*0.9))
    #print "sv is", sv
    #ind = np.arange(len(a))[sv]
    #print "index is", ind
    # Get the corresponding lagr. multipliers
    #a = a[sv]
    # Get the samples that will act as support vectors
    #sv_ = X[sv]
    #print sv_
    #print "support vectors", sv_, len(sv_)
    # Get the corresponding labels
    #sv_y = y[sv]
    #print sv_y
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
    
    """
    for n in range(len(a)):
        b += sv_y[n]
        b -= np.sum(a * sv_y * K[ind[n], sv])
    
    b=b/a
    # Weight vector
    w = np.zeros(X.shape[1])
    for n in range(len(a)):
        w += a[n] * sv_y[n] * sv_[n]
    """
    print("Learned Parameters - W:", w)
    #print len(w)
    print("Learned Parameters - B:", b[0][0])
    
    return w, b[0][0], a

#predicts the labels using w and b: w^t * x +b
def predict(w, b, X, C, alpha):
    print "in predict"
    y_pred=[]
    print w.shape, b.shape, X.shape
    w=w.reshape(w.shape[0],1)
    #b=b.reshape(b.shape[0],1)
    #b=b.reshape(X.shape[0],1)
    #print X.shape, w.shape, b.shape
    y=np.dot(X,w)+b
    y=np.ravel(y)
    #print y, len(y)
    for i in range(len(y)):
        #print y[i], " c is", C
        if int(np.sign(y[i]))==1:
            y_pred.append(1)
        else:
            y_pred.append(-1)
    return y_pred
    
#predict_test
def predict_test(w, b, X, C, alpha):
    y_pred=[]
    w=w.reshape(w.shape[0],1)
    #b=b.reshape(b.shape[0],1)
    print X.shape, w.shape, b.shape
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
    print "in acc"
    correctly_labelled=0 
    #print "true",y_true
    #print "pred",y_pred
        
    #np.place(y_true, y_true==-1, [0])
    for i in range(len(y_pred)):
        if int(y_pred[i])==int(y_true[i]):
            correctly_labelled=correctly_labelled+1
            #print "yay"
        
    print "correctly_labelled", correctly_labelled
    accuracy= float(float(correctly_labelled)/float(len(y_pred)))*100
    return accuracy


#predicts the labels using w and b and alpha: sum(alpha * y* kernel)+b
def predict_1(w, b, X, C, alpha, y_true):
    y_pred=[]
    #w=w.reshape(w.shape[0],1)
    #b=b.reshape(b.shape[0],1)
    #alpha=alpha.reshape(w.shape[0],1)
    K=apply_kernel('linear kernel', X)
    #no= alpha*y_true
    no=np.multiply(alpha, np.transpose(y_true))
    y=np.dot(no, K)+b
    print X.shape, w.shape, b.shape
    y=np.dot(X,w)+b
    y=np.ravel(y)
    #print y, len(y)
    for i in range(len(y)):
        #print y[i], " c is", C
        if y[i]>=C:
            y_pred.append(1)
        else:
            y_pred.append(0)
    print y_pred
    return y_pred
    
    
    
MAIN_1()
#MAIN_2()
#MAIN_3()
#MAIN_4()
#MAIN_5()
#MAIN_6()