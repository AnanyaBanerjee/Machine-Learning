#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 21:26:00 2018

@author: Mammu
"""

# assignment 1: promlem 1 : perceptron learning

import pandas as pd
import numpy as np
from sympy import symbols, diff

df=pd.read_csv("/Users/Mammu/Desktop/perceptron.csv")

print df.size, df.shape

df=df[0:999]

#print df

print df.columns
y_actual=list(df['y'])

#print target
W=[[0,0,0,0] for i in range(len(y_actual))]
B=[0 for i in range(len(y_actual))]
w_0=[0,0,0,0]
b_0=0
l=[]
#form vectors x=[x1, x2,x3,x4]
for i in range(0,df.shape[0]):
    x=[df["x1"][i], df["x2"][i],df["x3"][i], df["x4"][i]]
    l.append(x)

x=l  #array vector contining [x1, x2, x3, x4]

dW=[]
dB=[]

def perceptron_loss(w, b, x, y_actual):
    loss1=np.multiply(np.transpose(w),x)+b
    #print "loss1",loss1
    loss=np.sum(np.maximum(0, -y_actual*loss1))
    #print "loss inside loop", loss
    return loss

LOSS=[]

def positive(x):
    return x>0

def gradient_descent_2(w_0, b_0, y_actual, x, learning_rate, max_iter):
    w=w_0
    b=b_0
    i=0
    loss=0
    dw=0
    db=0
    while(i<max_iter):
        for j in range(len(y_actual)):
            loss=loss+perceptron_loss(W[i],B[i], x[j],y_actual[j])
            lo=np.multiply(np.transpose(w),x[j])+b
            lo1=-y_actual[j]*np.sum(lo)
            LOSS.append(lo1)
            if(lo1>=0):
              dw=-y_actual[j]*x[j]
              db=-y_actual[j]
            dW.append(dw)
            dB.append(db)
        #print "dw[i]", dW[i]
        if dW[i]==[]:
           dW[i]=[0]
        arr=np.array(dW[i])
        vb=learning_rate*arr
        vb1=list(vb)
        W[i]=np.add(W[i],vb1)
        B[i]=B[i]+learning_rate*dB[i]
        w=W[i]
        b=B[i]
        i=i+1
        if i==len(y_actual):
            print loss
            break;
    print "final loss is", loss
    return W, B





def stochastic_gradient_descent(w_0, b_0, y_actual, x, learning_rate, max_iter):
    w=w_0
    b=b_0
    i=0
    loss=1
    j=0
    while(i<max_iter):
        loss=0
        for j in range(len(y_actual)):
            loss=loss+perceptron_loss(W[j], B[j], x[j], y_actual[j])
            l0=np.sum(np.multiply(np.transpose(W[j]),x[j])+B[j])
            l1=np.multiply(-y_actual[j],l0)
            if l1>=0:
               W[j]=W[j]+learning_rate*np.multiply(y_actual[i],x[i])
               B[j]=B[j]+learning_rate*y_actual[i]
        i=i+1
        if i==len(y_actual):
            break
    print "final loss is", loss
    return W, B

    
    

#w1, b1=gradient_descent(w_0, b_0, y_actual, l, 1, 100)
#print "w is",len(w1)
#print "b is", len(b1)
w4, b4=gradient_descent_2(w_0, b_0, y_actual, l, 1, 100)
print "len(w) is",len(w4)
print "len(b) is", len(b4)
b4=sum(b4)

w2, b2=stochastic_gradient_descent(w_0, b_0, y_actual, l, 1, 100)
print "len(w) is",len(w2)
print "len(b) is", len(b2)
b2=sum(b2)

final_loss=list(filter(positive, LOSS))
sum_loss=np.sum(final_loss)
print "the gd loss is", sum_loss




