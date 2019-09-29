#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 18:06:07 2018

@author: Ananya Banerjee

Logistic Regression
"""


import numpy as np
from numpy import linalg as lg
import pandas as pd
import math
from cvxopt import matrix, solvers  #this package helps solve quadratic optimization problems
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

from sklearn.preprocessing import StandardScaler
#Accuracy Train
ACCURACY_train=[]

#train dataset
train_file=open("/Users/Mammu/Desktop/courses/machine learning/Assignment_5_ml/park_train.data")
dataset_train=np.loadtxt(train_file, dtype=np.dtype(float), delimiter=',')

#test dataset
test_file=open("/Users/Mammu/Desktop/courses/machine learning/Assignment_5_ml/park_test.data")
dataset_test=np.loadtxt(test_file, dtype=np.dtype(float), delimiter=',')

#train dataset
val_file=open("/Users/Mammu/Desktop/courses/machine learning/Assignment_5_ml/park_validation.data")
dataset_val=np.loadtxt(val_file, dtype=np.dtype(float), delimiter=',')


#validation test
df_val=dataset_val
df_val=np.delete(df_val,0,axis=1)
y_val=dataset_val[:,0]   
#df_val_projected=apply_pca(df_val, k, c, y_val)

#test set
df_test=dataset_test
df_test=np.delete(df_test,0,axis=1)
y_test=dataset_test[:,0]   
#df_val_projected=apply_pca(df_test, k, c, y_test)

#train
#t=range(0,61)
#dataset_train.columns=t
df=dataset_train
df=np.delete(df,0,axis=1)
#labels=dataset_train[::-1]
labels=dataset_train[:,0]



#replace all 0 with -1 in train, test and validation
for i in range(len(labels)):
    if labels[i]==0:
        labels[i]=-1
       
        
for i in range(len(y_val)):
    if y_val[i]==0:
        y_val[i]=-1
       
        
for i in range(len(y_test)):
    if y_test[i]==0:
        y_test[i]=-1


#sigmoid function
def sigmoid(z):
    return 1/float(1+np.exp(-z))


#function to find p(y=1/x) and p(y=-1/x)
def calculate_logit(w, b, x):
    v=np.exp((np.multiply(w.T,x)+b))
    p_1=float(v)/float(1+v)
    p_minus_1=1/float(1+v)
    return p_1, p_minus_1


#function to predict
def predict(p_1, p_minus_1):
    y_pred=[]
    for i in range(len(p_1)):
        if p_1[i]>=p_minus_1[i]:
            y_pred.append(1)
        else:
            y_pred.append(-1)
    return y_pred

#function to calculate accuracy
def find_accuracy(y_pred, y_true):
    count=0
    for i in range(len(y_pred)):
        if y_pred[i]==y_true[i]:
            count+=1
    
    acc= float(count)/float(len(y_true))
    return acc


#function to calculate loss function
def loss(h,y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

#function to predict probabilties
def predict_probs(X, theta):
    return sigmoid(np.dot(X, theta))

#function to predict
def predict(X, theta, threshold=0.5):
    return predict_probs(X, theta) >= threshold




#gradient = np.dot(X.T, (h - y)) / y.shape[0]
#lr = 0.01
#theta -= lr * gradient

model = LogisticRegression(C=1e5, solver='lbfgs')


# Create an instance of Logistic Regression Classifier and fit the data.
model.fit(df, labels)

y_pred=model.predict(df_test)

acc=metrics.accuracy_score(y_pred, y_test)
print "accuracy on test set is", acc

y_pred_val=model.predict(df_val)
acc1=metrics.accuracy_score(y_pred_val,y_val)
print "accuracy on val set is", acc1

#########ques 2: part2"#####: l2 penalty####
#preprocess the dataset



#dictionary containing C val as key and accuracy as val for l2 penalty
choice_of_c=dict()

#dictionary having weights and bias as val and key as value of c
parameters=dict()

#c is incerse of regularization strength
C=[0.000001, 0.00001, 0.0001,0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

for c in C:
    model1 = LogisticRegression(C=c, penalty='l2', solver='sag', random_state=0)
    
    # Create an instance of Logistic Regression Classifier and fit the data.
    model1.fit(df, labels)
    
    y_pred1=model1.predict(df_val)
    
    acc11=metrics.accuracy_score(y_pred1, y_val)
    print "accuracy on validation set using l2 is", acc11
    
    choice_of_c[c]=acc11
    
    w=model1.coef_
    #b=np.hstack((model.intercept_[:,None], model.coef_))
    b=model1.intercept_[:,None]
    parameters[c]=[w, b]


best_acc=np.max(list(choice_of_c.values()))
print "best accuracy using l2 penalty is", best_acc

chosen_c=0
for x in choice_of_c.keys():
    if(choice_of_c[x]==best_acc):
        chosen_c=x
        break;
    
print "chosen c using l2 penalty on validation set is", chosen_c

###testing it on test set#####
modeli = LogisticRegression(C=chosen_c, penalty='l2', solver='sag', random_state=0)
    
# Create an instance of Logistic Regression Classifier and fit the data.
modeli.fit(df, labels)
    
y_pred_test=modeli.predict(df_test)
ac=metrics.accuracy_score(y_pred_test, y_test)
print "accuracy on test set using chosen c is", ac




############ques 2: part 3::::l1 penalty
    
#dictionary containing C val as key and accuracy as val for l1 penalty
choice_of_c_t=dict()

#dictionary having weights and bias as val and key as value of c
parameters_t=dict()

#c is incerse of regularization strength
C_t=[0.0000001,0.00001, 0.0001,0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

for c in C_t:
    model2 = LogisticRegression(C=c, penalty='l1',random_state=0)
    
    # Create an instance of Logistic Regression Classifier and fit the data.
    model2.fit(df, labels)
    
    y_pred1=model2.predict(df_val)
    
    acc11=metrics.accuracy_score(y_pred1, y_val)
    print "accuracy on validation set using l1 is", acc11
    
    choice_of_c_t[c]=acc11
    
    w=model2.coef_
    #b=np.hstack((model.intercept_[:,None], model.coef_))
    b=model2.intercept_[:,None]
    parameters_t[c]=[w, b]



best_acc_t=np.max(list(choice_of_c_t.values()))
print "best accuracy using l1 penalty is", best_acc_t

chosen_c_t=0
for x in choice_of_c_t.keys():
    if(choice_of_c_t[x]==best_acc_t):
        chosen_c_t=x
        break;
    
print "chosen c using l1 penalty on validation set is", chosen_c_t

###testing it on test set#####
modelit = LogisticRegression(C=chosen_c_t, penalty='l1',  random_state=0)
    
# Create an instance of Logistic Regression Classifier and fit the data.
modelit.fit(df, labels)
    
y_pred_test1=modelit.predict(df_test)
ac1=metrics.accuracy_score(y_pred_test1, y_test)
print "accuracy on test set using chosen c is", ac1
    

    
    
