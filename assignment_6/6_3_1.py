#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 23:06:35 2018

@author: Ananya Banerjee

Neural Network
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
#from sklearn.neural_network import MLPCLassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import math

#train dataset
df=pd.read_csv("/Users/Mammu/Desktop/courses/machine learning/Assignment_6/data_set_generated.csv", delimiter=None)

target=df["y"]

del df["y"]

#function to initialize weights using randomly, following standard normal distribution 
def initialize_weight_random(size):
    #w=np.random.randn(size, size-1)
    w=np.random.randn(size, 1)
    return w


#function to initialize bias
def initialize_bias(size):
    b=np.random.randn(1)
    return b


#function to implement simple perceptron
def find_weighted_sum(w, b, x):
    w=w.T
    weighted_sum=np.dot(w, df)
    weighted_sum=[i+b for i in weighted_sum]
    #weighted_sum has shape( n , 1)
    return weighted_sum

#sigmoid function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

#ReLu function
def ReLU(x):
   return np.maximum(x, 0)


#function to implement sigmoid activation function
def activation_func_sigmoid(weighted_sum, type_of_func):
   activations=[] #list of activation
   #if activation function used if sigmoid
   if type_of_func=="sigmoid":
      for i in range(len(weighted_sum)):
          a=sigmoid(weighted_sum[i])
          activations.append(a)
    
   if type_of_func=="relu":
      for i in range(len(weighted_sum)):
          a=ReLU(weighted_sum[i])
          activations.append(a)
    
   return activations

  
    
#Stochastic Gradient Descent
def perceptron_sgd_plot(X, Y):
    '''
    train perceptron and plot the total loss in each epoch.
    
    :param X: data samples
    :param Y: data labels
    :return: weight vector as a numpy array
    '''
    w = np.zeros(len(X[0]))
    eta = 1
    n = 30
    errors = []

    for t in range(n):
        total_error = 0
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                total_error += (np.dot(X[i], w)*Y[i])
                w = w + eta*X[i]*Y[i]
        errors.append(total_error*-1)
        
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    
    return w
    
#cross entropy loss function
def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    p = softmax(X)
    # We use multidimensional array indexing to extract 
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss

#function to implement perceptron
def perform_perceptron(X_train,y_train, X_test, y_test):
    clf = Perceptron(tol=1e-3, random_state=0)
    model=clf.fit(X_train, y_train)
    print "Model score for perceptron", model.score(X_train, y_train)
    print "parameters for perceptron are", model.get_params
    y_pred=model.predict(X_test)
    score=accuracy_score(y_pred, y_test)
    print "The accuracy is", score
    
  

#function to implement MLP
def perform_MLP(X_train,y_train, X_test, y_test):
    clf = MLPClassifier(activation='relu', solver='adam',momentum=0.9,  )
    model=clf.fit(X_train, y_train)
    print "Model score for mLP is", model.score(X_train, y_train)
    print "parameters for MLP are", model.get_params
    y_pred=model.predict(X_test)
    score=accuracy_score(y_pred, y_test)
    print "The accuracy for MLP is", score
    
    
    
#function to perform train-test split
def perform_train_test_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return  X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test=perform_train_test_split(df,target)
perform_perceptron(X_train,y_train, X_test, y_test)
perform_MLP(X_train,y_train, X_test, y_test)


siz=df.shape[0]
l=initialize_weight_random(siz)

  