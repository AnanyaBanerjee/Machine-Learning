#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 16:07:09 2018

@author: Ananya Banerjee : AdaBoost using decision trees
"""

import numpy as np
import pandas as pd
import math

#train dataset
dataset_train=[]

#test dataset
dataset_test=[]

#dictionary containing split datasets
GD=dict()

#dictionary of predictions of all iterations
PRED=dict()

#target
target='OVERALL_Diagnosis'

#original target col
target_col=[]

#predicted target col
predicted_col=[]

#dictionary to keep track of alpha as key and corrsponding h_m(x) as values
H=dict()

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
    df1=df[df[target]==0]
    df2=df[df[target]==1]
    if(len(df1[target])==0 or len(df2[target])==0):
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
    df1=df[df[target]==0]
    df2=df[df[target]==1]
    
    #returns 0 if majority vote says label is 0 and 1 ow
    if(len(df1[target])> len(df2[target])):
        return 0
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
    
    
#function to test on test set
def test(w, predictions, list_of_col, attr1, attr2, attr3):
    
    #making a dataframe for test
    df_test=pd.DataFrame(dataset_test, columns=list_of_col)
    df_test=df_test.fillna(0)
    
    #predictions: labels corresponding to:
    #[000, 001, 010, 011, 100, 101, 110, 111]#
    
    df_test[str(target)+'pred']=[0.0 for j in range(len(df_test[target]))]
   
    for i in range(len(df_test[target])):
        if df_test[attr1].loc[i]==0 and df_test[attr2].loc[i]==0:
            if df_test[attr3].loc[i]==0: #000
                df_test[str(target)+'pred'].loc[i]=predictions[0]
            else:  #001
                df_test[str(target)+'pred'].loc[i]=predictions[1]
        
        if df_test[attr1].loc[i]==0 and df_test[attr2].loc[i]==1:
            if df_test[attr3].loc[i]==0: #010
                df_test[str(target)+'pred'].loc[i]=predictions[2]
            else:  #011
                df_test[str(target)+'pred'].loc[i]=predictions[3]
        
        if df_test[attr1].loc[i]==1 and df_test[attr2].loc[i]==0:
            if df_test[attr3].loc[i]==0: #100
                df_test[str(target)+'pred'].loc[i]=predictions[4]
            else:  #101
                df_test[str(target)+'pred'].loc[i]=predictions[5]
        
        if df_test[attr1].loc[i]==1 and df_test[attr2].loc[i]==1:
            if df_test[attr3].loc[i]==0: #110
                df_test[str(target)+'pred'].loc[i]=predictions[6]
            else:  #111
                df_test[str(target)+'pred'].loc[i]=predictions[7]
        
    acc=find_accuracy(df_test[target], df_test[str(target)+'pred'])
    err=calc_error(w, df_test[str(target)+'pred'], df_test[target])
    #print "accuracy is", acc
    return acc, err
 
#function to test on test set
def test_train(w, predictions, list_of_col, attr1, attr2, attr3, df):
    
    #predictions: labels corresponding to:
    #[000, 001, 010, 011, 100, 101, 110, 111]#
    
    df[str(target)+'pred']=[0.0 for j in range(len(df[target]))]
   
    for i in range(len(df[target])):
        if df[attr1].loc[i]==0 and df[attr2].loc[i]==0:
            if df[attr3].loc[i]==0: #000
                df[str(target)+'pred'].loc[i]=predictions[0]
            else:  #001
                df[str(target)+'pred'].loc[i]=predictions[1]
        
        if df[attr1].loc[i]==0 and df[attr2].loc[i]==1:
            if df[attr3].loc[i]==0: #010
                df[str(target)+'pred'].loc[i]=predictions[2]
            else:  #011
                df[str(target)+'pred'].loc[i]=predictions[3]
        
        if df[attr1].loc[i]==1 and df[attr2].loc[i]==0:
            if df[attr3].loc[i]==0: #100
                df[str(target)+'pred'].loc[i]=predictions[4]
            else:  #101
                df[str(target)+'pred'].loc[i]=predictions[5]
        
        if df[attr1].loc[i]==1 and df[attr2].loc[i]==1:
            if df[attr3].loc[i]==0: #110
                df[str(target)+'pred'].loc[i]=predictions[6]
            else:  #111
                df[str(target)+'pred'].loc[i]=predictions[7]
    
    target_col.append(df[target]) 
    predicted_col.append(df[str(target)+'pred'])
    #print target_col
    #print predicted_col
    acc=find_accuracy(df[target], df[str(target)+'pred'])
    err=calc_error(w, df[str(target)+'pred'], df[target])
    #print "accuracy is", acc
    return acc, err

    
#function to fit decision trees
def make_decision_tree(w,attr1, attr2, attr3, df, itr, list_of_col):
    #print df[attr1], type(df[attr1].loc[0])
    #split the dataset using attr1
    g=np.float64(0)
    h=np.float64(1)
    df1_1=df[df[attr1]==g] #attr1=0
    df1_2=df[df[attr1]==h] #attr1=1
    #split the dataset using attr2
    df2_1=df1_1[df1_1[attr2]==g] #attr1=0, attr2=0
    df2_2=df1_1[df1_1[attr2]==h] #attr1=0, attr2=1
    df2_3=df1_2[df1_2[attr2]==g] #attr1=1, attr2=0
    df2_4=df1_2[df1_2[attr2]==h] #attr1=1, attr2=1
    #split the dataset using attr3
    df3_1=df2_1[df2_1[attr2]==g] #attr1=0, attr2=0, attr3=0
    df3_2=df2_1[df2_1[attr2]==h] #attr1=0, attr2=0, attr3=1
    df3_3=df2_2[df2_2[attr2]==g] #attr1=0, attr2=1, attr3=0
    df3_4=df2_2[df2_2[attr2]==h] #attr1=0, attr2=1, attr3=1
    df3_5=df2_3[df2_3[attr2]==g] #attr1=1, attr2=0, attr3=0
    df3_6=df2_3[df2_3[attr2]==h] #attr1=1, attr2=0, attr3=1
    df3_7=df2_4[df2_4[attr2]==g] #attr1=1, attr2=1, attr3=0
    df3_8=df2_4[df2_4[attr2]==h] #attr1=1, attr2=1, attr3=1
    
    #list of split datasets
    list_of_split_datasets=[ df3_1, df3_2, df3_3, df3_4, df3_5, df3_6, df3_7, df3_8]
    #dictionary containing split datasets where key is iteration number
    ab=dict()
    
    ab[itr]=list_of_split_datasets
    
    #add this to global datasets
    GD[itr]=ab
    
    #list of predcitions
    pred=[]
    
    for x in list_of_split_datasets:
         #now check if the datasets are pure or not, if they are assign labels
         pure=check_purity(x)
         if(pure==1):
             empty=is_null(x) #if this is 1, says its null
             if empty==1:
                 pred.append(0)
                 continue
             #print x[target], target
             #dataset is pure, then assign that label as predicted label
             c=list(x[target])[0]
             #pred[str(x)]=c
             pred.append(c)
         #if not pure then use majority vote
         else:
            empty=is_null(x) #if this is 1, says its null
            if empty==1:
                pred.append(0)
                continue
            vote=majority_vote(x)
            #pred[str(x)]=vote
            pred.append(vote)
    
    PRED[itr]=pred
    acc, err=test_train(w, pred, list_of_col, attr1, attr2, attr3, df)
    print "accuracy is", acc
    print "error is", err
    return err
        
#function to find error
def calc_error(w, pred, orig):
    sum1=0.0
    #print len(w), len(pred), len(orig)
    for i in range(len(orig)):
        if(pred[i]!=orig[i]):
            sum1=sum1+w[i]
    return sum1
       
#function to update weights
def update_weights(err, alpha, w, itr):
    #print "tc", target_col
    #print "pc", predicted_col
    t=target_col[itr] #contains orig target col
    p=predicted_col[itr] #contains pred col
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


#function to perform adaboost 
def AdaBoost(df, max_iter,list_of_col):
    M=len(df[target])
    #initialize weights 
    df[target].loc[80]=0.0
    w=[float(1/float(M)) for i in range(len(df[target]))]
    
    for itr in range(max_iter):
        l=list(df.columns.to_series().sample(3))
        print "chosen attributes are:", l
        err=make_decision_tree(w, l[0], l[1], l[2], df, itr, list_of_col)
        #find alpha_t which indicates how much value you give to your hypothesis
        ert=float(1-err)/float(err)
        alpha_t=0.5*math.log(ert)
        #update the weights
        w=update_weights(err, alpha_t, w, itr)
    
    
    
    
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
    
    #df_i=df
    #del df_i[target]
    """
    for itr in range(3):
       #pick random three attrbutes
       l=list(df.columns.to_series().sample(3))
       print "chosen attributes are:", l
       make_decision_tree(l[0], l[1], l[2], df, itr, list_of_col)
    
    """
    AdaBoost(df, 3, list_of_col)
    #now we find the weighted sum to find final Hypothesis H
    h_final=0.0
    for x in list(H.keys()):
        h_final=h_final+x*H[x]
    
    print "final hypothesis is", h_final
    for i in range(len(h_final)):
        if(h_final[i]<0.5):
            h_final[i]=1
        else:
            h_final=0
    
    print "finale", h_final
    acc=find_accuracy(df[target], h_final)
    print "Accuracy for final hypothesis is", acc
    
MAIN()