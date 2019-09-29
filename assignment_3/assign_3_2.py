#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 13:53:32 2018

@author: Mammu
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:55:39 2018

@author: Ananya
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 16:07:09 2018

@author: Ananya Banerjee : AdaBoost using decision trees
"""

import numpy as np
import pandas as pd
import math
import random

#train dataset
dataset_train=[]

#test dataset
dataset_test=[]

#dictionary containing split datasets
GD=dict()

#dictionary of predictions of all iterations
PRED=dict()

#dictionary of first choice of attributes as key and corresponding errors as values
CHOICE=dict()

#dictionary containing err as keys and hypothesis as values
Hypothesis=dict()

#target
target='OVERALL_Diagnosis'

#original target col
target_col=[]

#predicted target col
predicted_col=[]

#dictionary to keep track of alpha as key and corrsponding chosen h_m(x) as values
H=dict()

#list of all 2^4 values
label_combo=[0000,0001,0010,0011,0100,0101,0110,0111, 1000,1001,1010,1011,1100,1101,1110,1111]


def change_sign(inp):
    if inp==0:
        return -1
    else:
        return 1

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
     
        else:
           dataset_test.append(lines) 
        # print len(lines), lines
        if not line:
            break
    #print dataset_train
    #print "hey",len(dataset_train), len(dataset_train[0])
   
#function to check for purity
def check_purity(df):
    #since target has only 0 and 1
    df1=df[df[target]==-1]
    df2=df[df[target]==1]
    if(len(df1[target])==-1 or len(df2[target])==1):
        #that is if you have only 0's or 1's
        #print "pure dataset detected"
        purity=1  #if pure, return 1
    else:
        purity=0 #else 0
       
    return purity

#function to check if a dataset is null or not
def is_null(df):
    
    #print "inside is_null"
    if(df.empty):
        return 1 #1 means dataset is null
    else:
        return 0 #0 means none of the datasets are null
 

#function to use majority vote
def majority_vote(df):
    #since target has only 0 and 1
    df1=df[df[target]==-1]
    df2=df[df[target]==1]
    
    #returns 0 if majority vote says label is 0 and 1 ow
    if(len(df1[target])> len(df2[target])):
        return -1
    else:
        return 1
   
#function to find accuracy
def find_accuracy(orig, pred):
    count=0;
    for i in range(len(orig)):
        if(orig[i]==pred[i]):
                count+=1
    #print "count is", count
    accuracy=float(float(count) / float(len(orig)) )* 100
    return accuracy
    
        
#function to find error
def calc_error(w, pred, orig):
    sum1=0.0
    #print len(w), len(pred), len(orig)
    for i in range(len(orig)):
        if(pred[i]!=orig[i]):
            sum1=sum1+w[i]
    return sum1
       
#function to update weights
def update_weights(predicted_col, err, alpha, w, itr):
    #print "tc", target_col
    #print "pc", predicted_col
    t=target_col[0] #contains orig target col
    p=predicted_col#[itr] #contains pred col
    #update H
    H[alpha]=p
    for i in range(len(w)):
        #print "it is", i, t[i], p[i], len(t), len(p)
        et=np.exp(-t[i]*p[i]*alpha)
        ed=2*np.sqrt(err*(1-err))
        if i!=80:
           w[i+1]=float(w[i]*et)/ float(ed)
        #print w[i]
    return w

#function to apply tree of height 1
"""
        attr1
    0   -   -  1
    -         -
-1               1
"""
def apply_tree_1(df, attr1,w):
    li_of_hp=[]
    for i in range(len(df[target])):
        if(df[attr1].loc[i])==0:
            li_of_hp.append(-1.0)
        else:
            li_of_hp.append(1.0)
    err=calc_error(w, li_of_hp, df[target])
    acc=find_accuracy(df[target], li_of_hp)
    print "err is, ",err, " and acc is", acc
    return err, li_of_hp
 
#function to apply tree of height 1
"""
        attr1
    0   -   -  1
    -         -
1                -1
"""
def apply_tree_2(df, attr1,w):
    li_of_hp=[]
    for i in range(len(df[target])):
        if(df[attr1].loc[i])==0:
            li_of_hp.append(1.0)
        else:
            li_of_hp.append(-1.0)
    err=calc_error(w, li_of_hp, df[target])
    acc=find_accuracy(df[target], li_of_hp)
    print "err is, ",err, " and acc is", acc
    return err, li_of_hp
  
#function to apply tree of height 1
"""
        attr1
    0   -   -  1
    -         -
-1                -1
"""
def apply_tree_3(df, attr1,w):
    li_of_hp=[]
    for i in range(len(df[target])):
        if(df[attr1].loc[i])==0:
            li_of_hp.append(-1.0)
        else:
            li_of_hp.append(-1.0)
    err=calc_error(w, li_of_hp, df[target])
    acc=find_accuracy(df[target], li_of_hp)
    print "err is, ",err, " and acc is", acc
    return err, li_of_hp

#function to apply tree of height 1
"""
        attr1
    0   -   -  1
    -         -
1                1
"""
def apply_tree_4(df, attr1,w):
    li_of_hp=[]
    for i in range(len(df[target])):
        if(df[attr1].loc[i])==0:
            li_of_hp.append(1.0)
        else:
            li_of_hp.append(1.0)
    err=calc_error(w, li_of_hp, df[target])
    acc=find_accuracy(df[target], li_of_hp)
    print "err is, ",err, " and acc is", acc
    return err, li_of_hp

#function to form hypothesis 
def find_hypothesis(df):
    l=list(df.columns)
    l=l[1:]
    w=[float(1)/float(df.shape[1]) for i in range(len(df[target]))]
    #make hypothesis trees of 1 height each
    for x in l:
        print "x is",x
        err1, pred1=apply_tree_1(df, x,w)
        err2, pred2=apply_tree_2(df, x,w)
        err3, pred3=apply_tree_3(df, x,w)
        err4, pred4=apply_tree_4(df, x,w)
        Hypothesis[err1]=[pred1]
        Hypothesis[err2]=[pred2]
        Hypothesis[err3]=[pred3]
        Hypothesis[err4]=[pred4]
        
        
#function to find alpha
def compute_alpha(y, y_pred, alpha_i, t):
    #
    sum_n=0.0 #sum numerator
    sum_d=0.0 #sum denominator
    sum_o=0.0 #this sum -> -v
    #find sum of all alpha_i * h_t for t
    for i in range(len(list(Hypothesis.keys()))):
        #print i,alpha_i[i], y_pred[i], len(y_pred), len(alpha_i)
        if i!=t:
           sum_o=sum_o+np.sum(np.multiply(alpha_i[i], y_pred[i]))
    
    
    for i in range(len(y)):
        if y[i]==y_pred[i]:
            sum_n=sum_n+np.exp(-y[i]*sum_o)
        else:
            sum_d=sum_d+np.exp(-y[i]*sum_o)
        
    final_alpha=0.5*(float(sum_n)/float(sum_d))
    return final_alpha

#function to perform coordinate descent
def coordinate_descent(df, max_iter):
    alpha_i=[float(1)/float(df.shape[1]) for i in range(len(df[target]))]
    for i in range(max_iter):
        t=random.choice(Hypothesis.keys())
        y_pred=Hypothesis[t]
        #print "y is", y_pred
        alpha_updated=compute_alpha(df[target], y_pred[0], alpha_i, t)
        H[alpha_updated]=y_pred[0]
    
    #print "Final alpha is", alpha_updated
    #return alpha_updated
   
#function to perform coordinate descent
def coordinate_descent_1(df, max_iter):
    alpha_i=[float(1)/float(df.shape[1]) for i in range(len(df[target]))]
    j=0
    while(alpha_i[j]!=0):
        t=random.choice(Hypothesis.keys())
        y_pred=Hypothesis[t]
        #print "y is", y_pred
        alpha_updated=compute_alpha(df[target], y_pred[0], alpha_i, t)
        H[alpha_updated]=y_pred[0]
    
    #print "Final alpha is", alpha_updated
    #return alpha_updated
      
     
#function MAIN
def MAIN():
    
    #open train data
    file_name_train="/Users/Mammu/Desktop/courses/machine learning/assignment_3/heart_train.data"
    #open test file
    open_file(file_name_train,'train')
    
    file_name_test="/Users/Mammu/Desktop/courses/machine learning/assignment_3/heart_test.data"
    #open test file
    open_file(file_name_test,'test')
    
    #list of columns
    list_of_col=[]
    
    #print dataset_train, type(dataset_train), type(dataset_train[0])
                        
    for i in range(len(dataset_train[0])):
        if i==0:
            list_of_col.append('OVERALL_Diagnosis')
        else:
            list_of_col.append('F'+str(i)+": 0,1")
   
    print list_of_col
    
    #making a dataframe
    df=pd.DataFrame(dataset_train, columns=list_of_col)
    df=df.fillna(0)
    for i in range(len(df[target])):
        if df[target].loc[i]==0:
            df[target].loc[i]=-1
    
      
    df[target].loc[80]=-1
    target_col.append(df[target])
    
    #df_i=df
    #del df_i[target]
    find_hypothesis(df)
    #AdaBoost(df, 1, list_of_col)
    coordinate_descent(df, 20000)
    #calculate the loss function
    lo=0.0
    #j=0
    loss=0.0
    for j in range(df.shape[1]):
        for i in list(H.keys()):
            pred=H[i]
            lo+=np.sum(np.multiply(i, pred[j]))
        loss+=np.exp(-df[target].loc[j]*lo)
    
    print "the Loss is", loss
    
    for b in H.keys():
        pred=H[b]
        acc1=find_accuracy(df[target], pred)
        print "accuracy for alpha", b, "is", acc1
    
    #final hypothesis is given by sum(h*alpha)
    sum1=[0 for i in range(len(df[target]))]
    for x in H.keys():
        H[x]=[i*x for i in H[x]]
        sum1=[x + y for x, y in zip(H[x], sum1)]
        
    #print sum1
    final_h=[]
    for x in sum1:
        if(np.sign(x)):
            final_h.append(1.0)
        else:
            final_h.append(-1.0)
            
    acc=find_accuracy(df[target], final_h)
    print "Accuracy on training set is", acc
    
MAIN()