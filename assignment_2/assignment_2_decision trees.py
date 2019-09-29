#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 14:28:03 2018

@author: Ananya Banerjee

Assignment 2: Problem 2: Decision Trees
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy
import math

df_train=pd.read_csv("/Users/Mammu/Desktop/mush_train.csv")

df_test=pd.read_csv("/Users/Mammu/Desktop/mush_test.csv")

TREE=dict() #this contains the entire tree, each node containing GD

#global dictionary containing split names as keys and the corresponding splitted datasets as values
GD=dict()

list_of_col_names=['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root','stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat' ]
#print len(list_of_col_names)
    
df_train.columns=list_of_col_names

#target values
target1=list(df_train['target'])


#make_dictionary('odor',gh)
decision_null=0 # this tells us whether we got any nodes containing empty datasets
decision_purity=0; #this tells is whether we got any pure dataset in nodes
   


def make_dictionary(col_name, df):
    
    print "inside make_dictionary"
    #col_name is name of the column that was split
    #df is a list of lists that contains the split datasets
    #Goal: To make a dictionary, containing all splits
    for i in range(len(df)):
        name=str(col_name)+"_split_"+str(i)  #GD={'odor_split_1': dataset1, ...}
        GD[name]=df[i]
    
    TREE[col_name]=GD
    #print GD
 
    
def find_cond_entropy_1(dataset, entropy_y):
    print "inside entropy is", 
    #target col
    tar=dataset['target']
    tar_classes=np.unique(tar)
    #list of IG
    IG=[]
    for x in dataset.columns:
        if x=='target':
            IG.append(-1000) #just append a random value of H_y
            continue
        #print "col is", x
        #partition the dataset based on target values
        #df1=dataset[dataset['target']=='e']
        #df2=dataset[dataset['target']=='p']
        #find classes of the col x
        no_of_classes=list(np.unique(list(dataset[x]))) #gives the number of unique classes
        #print "no_of_classes", no_of_classes
        #for every class, form a dataset and append it to this list
        class_d=[]
        #list of probablity p(Y=y/X=x)
        P_y_x=[]
       
        #probability of the class occuring in a column : P(X=x)
        prob_x=float(1/float(len(no_of_classes)))
        
        for j in no_of_classes:
            #print "class is", j
            #dataset containing class j of col x
            dfi=dataset[dataset[x]==j]
            
            #dfi_1 contains the dataset having class j in col x and target e
            dfi_1=dfi[dfi['target']=='e']
            
            #dfi_2 contains the dataset having class j in col x and target p
            dfi_2=dfi[dfi['target']=='p']
            
            #find probability
            #prob of class j having target e
            prob_y_x_1=float(float(dfi_1.shape[0])/float(dfi.shape[0]))
            #prob of class j having target p
            prob_y_x_2=float(dfi_2.shape[0])/float(dfi.shape[0])
            
            if prob_y_x_1!=0:
               prob_y_x_1=-1*prob_y_x_1 * math.log(prob_y_x_1,2)
            else:
               prob_y_x_1=0.0
           
            if prob_y_x_2!=0:
               prob_y_x_2=-1*prob_y_x_2 * math.log(prob_y_x_2,2)
            else:
               prob_y_x_2=0.0
            #print prob_y_x,"prob", float(dfi_1.shape[0])/float(dfi.shape[0]), float(dfi_2.shape[0])/float(dfi.shape[0])
            #print prob_y_x, "hey"
            prob_y_x=prob_y_x_1+prob_y_x_2
            P_y_x.append(prob_y_x)
            
            class_d.append(dfi)
        
        #print "prob of len of lcass", prob_x
        sum1=float(np.sum(P_y_x)*prob_x)
        
        #sum1=-1*sum1
        print "H(Y/X) for X=",x," is", sum1
        ig=find_information_gain(entropy_y, sum1)
        IG.append(ig)
    
    #print IG
    return IG
    

    
#function to find H(y)
def find_entropy_y(target_1):
    print "inside entropy_y"
    #j=0
    target_1=list(target_1)
    t=list(np.unique(target_1))#['e','p']
    print " t is", t
    H_y=0
    
    count_e=0
    count_p=0
    for i in range(len(target_1)):
        if target_1[i]=='e':
            count_e+=1
        else:
            count_p+=1
    
    c_1= float(count_e)/float(len(target_1))
    g_1= -1*c_1* math.log(c_1,2)
    c_2= float(count_p)/float(len(target_1))
    g_2= -1*c_2* math.log(c_2,2)
    
    H_y=g_1+g_2
    
    print "Entropy wrt H(y)", H_y
    
    
    return H_y

# function to split the dataset
def split_dataset(col_name, col_to_split_on, dataset):
    print "inside split "
    #print col_name, col_to_split_on
    #no_of_classes contains the nodes that we need to split the dataset in
    no_of_classes=list(np.unique(col_to_split_on))
    
    #print "classes is", no_of_classes
    #print no_of_classes
    #list of dataset split datasets
    df=[]
    #we need to create datasets for each class of the attribute
    for i in no_of_classes:
        dfi=df_train[df_train[col_name]==i]
        df.append(dfi)
    
    return df
    

#function to find information gain
def find_information_gain(H_y, H_y_x):
    IG= H_y - H_y_x
    return IG



#function to check for purity
def check_purity(df):
    #df is dataset
    print "inside purity"
    #if type(df)!=list:
    #    return 0
    
    #df is list of datasets
    #target col
    #tc=list(np.unique(target))
    tar=list(df['target'])
    #t=list(np.unique(tar))
    count_e=0
    count_p=0
    for i in range(df.shape[0]):
        if tar[i]=='e':
            count_e+=1
        else:
            count_p+=1
    print "for target, p:", count_p, "and e is:", count_e
    if count_e==0 and count_p==len(tar):
        #pure dataset in p
        purity=1
        print "pure dataset detected"
    elif count_p==0 and count_e==len(tar):
        #pure dataset in e
        purity=1
        print "pure dataset detected"
    else:
        purity =0
        
    """
    #check if dataset is pure or not
    ent=find_entropy_y(df['target'])  #check if entropy of y is 0, if it is then the dataset is pure
    if(ent<=0.0):
       print "pure dataset detected"
       purity=1  #if pure, return 1
    else:
        purity=0 #else 0
    """   
    return purity

#function to check if a dataset is null or not
def is_null(df):
    
    print "inside is_null"
    if type(df)!=list:
        return 0
    #no_of_datasets contains the number of splitted datasets
    no_of_datasets=len(df)
    
    #count tells us whether the dataset after split is null or not
    count=0
    
    data=0
    
    #print "df is",list(df)
    #print "df is", type(df)
    for i in range(no_of_datasets):
        data=df[i] #get the dataset from the list
        if(data.empty):
           count=count+1
           #print "This is null"
           
    if(count>0):
        return 1 #1 means dataset is null
    else:
        return 0 #0 means none of the datasets are null
    

#############################################
      
    

#function to build a tree--finds best attribute and splits and returns the splitted datasets and the col on which the split was done
def build_tree(dataset):
    
    print "inside build_tree"
    
    #extract the columns in a list
    list_of_col=list(dataset.columns)

    print "list of col for thie dataset is", list_of_col
    #initialise h_y=0 and h_y_x=0
    h_y=0
    #h_y_x=0
   
    h_y=find_entropy_y(dataset['target'])
    IG=find_cond_entropy_1(dataset, h_y)
    
    #now IG contains all the Information gains for each attributes
    maxi=np.max(IG)  #contains max value of IG
    max_ind=IG.index(maxi)    #contains index of max value
    
    print "The attribute with highest IG is", list_of_col[max_ind], "and the value is", maxi
    
    print "You should split on col", list_of_col[max_ind]
    
    list_of_split_datasets=split_dataset(str(list_of_col[max_ind]), dataset[list_of_col[max_ind]], dataset)
    
    print "number of splits is", len(list_of_split_datasets)
    
    return list_of_split_datasets, str(list_of_col[max_ind])
    
    
    

################################################
   

        
###################################################
    

#gh=make_tree(df_train, target)




#####recursive function to perform splits till you have traversed all the attributes
#col==col to split on
#col_name== name of the column that needs splitting
#target== target column
#dataset==list of datasets
def perform(dataset, target):
    print "Inside perform"
    #print dataset
    #if the dataset just has one column then you have reached leaf node
    if len(list(dataset.columns))==1:
        print "i am done"
        return
    
    #if the dataset is null, return
    if is_null(dataset):
        return
        
    #if the dataset made is pure, u r done
    if check_purity(dataset)==1:
        return

    #else you build the tree
    #DF is the list containing all split datasets
    DF, col_name=build_tree(dataset)
    print "the col name with highest ig is", col_name
    
    for x in DF:
        print "col to be dropped is", col_name
        #x=x.drop([col_name], axis=1)
        #print x.columns
        del x[col_name]
        
        
    #add these splits into the dictionary
    make_dictionary(col_name, DF)
       
    #print "DF is", DF[0].shape, col_name, DF[0].columns
    for i in range(len(DF)):
        print "i th dataset is going in", i
        #now delete the col from datatset that you are sending next
        #del DF[i][col_name]
        print "the target is", np.unique(DF[i]['target'])
        #print "the col are", DF[i].columns
        #check for purity and only send those datasets that need splitting
        if check_purity(DF[i])==1:
            continue;
        else:
           #perform(DF[i], target)#, DF[i][col_name], col_name)
           print "i need splitting"
           return perform(DF[i], target)


def perform_1(dataset, target, t):
    print "Inside perform"
    if len(t)==1:
        return
    
    #if the dataset just has one column then you have reached leaf node
    if len(list(dataset.columns))==1:
        return
        
    #if the dataset is null, return
    if is_null(dataset):
        return
        
    #if the dataset made is pure, u r done
    if check_purity(dataset)==1:
        return

    else:
        if len(t)==2:
            #you have 2 classes and you need to split the dataset
            #else you build the tree
            #DF is the list containing all split datasets
            DF, col_name=build_tree(dataset)
            print "the col name with highest ig is", col_name
            #add these splits into the dictionary
            make_dictionary(col_name, DF)
    
            #print "DF is", DF[0].shape, col_name, DF[0].columns
            for i in range(len(DF)):
                print "i th dataset is going in", i
                #now delete the col from datatset that you are sending next
                del DF[i][col_name]
                t=np.unique(DF[i]['target'])
                print "the target is", t
                #print "the col are", DF[i].columns
                #check for purity and only send those datasets that need splitting
                if check_purity(DF[i])==0:
                    continue;
                else:
                    perform_1(DF[i], target,t)#, DF[i][col_name], col_name)
        

        #print "Inside perform"
        #print dataset
        
        #else you build the tree
        #DF is the list containing all split datasets
        DF, col_name=build_tree(dataset)
        print "the col name with highest ig is", col_name
        #add these splits into the dictionary
        make_dictionary(col_name, DF)
    
        #print "DF is", DF[0].shape, col_name, DF[0].columns
        for i in range(len(DF)):
            print "i th dataset is going in", i
            #now delete the col from datatset that you are sending next
            del DF[i][col_name]
            t=np.unique(DF[i]['target'])
            print "the target is", t
            #print "the col are", DF[i].columns
            #check for purity and only send those datasets that need splitting
            if check_purity(DF[i])==0:
                continue;
            else:
                perform_1(DF[i], target,t)#, DF[i][col_name], col_name)
        

"""    
#function to call perform
def f(dataset, target):
    print "inside f"
    for i in list(dataset.columns):
        #find the attribute with the highest gain    
        perform(dataset, target)

    #perform(dataset,target)     
"""

perform(df_train, target1)   
#perform_1(df_train, target1, [])





"""
#en_x=find_entropy_x(df_train['f'], target, 'f')
en_y=find_entropy_y(target)

#list containing H(y/x) for x where x stands for attributes of the dataset
H_X=[]
#info gain list for all attributes 
IG=[]

for i in list(df_train.columns):
    #i is value of columns
    if i=='e':
        #it means 1st column is target
        en_X=0
    else:
        en_X=find_entropy_x(df_train[i], target, i)
    
    #find information gain for each attribute
    ig= en_y- en_X
    print "Info Gain for", i, "is", ig
    H_X.append(en_X)
    IG.append(ig)


print "Information gain is", IG
    
#list of datasets splits
df=split_dataset('odor', df_train['odor'], df_train)
  
    
 """  
    
    
    
    
    
    
    
    
    
    
    
    
    

"""
j=0
t=list(np.unique(target))#['e','p']

H_y=0

for i in range(len(t)):
    df1=df_train[df_train['e']==t[j]]
    j=j+1
    c=(float(df1.shape[0])/float(len(target)))
    print c
    g=-1*c*np.log(c)
    H_y=H_y+g
    #print len(df1)

print "Entropy wrt H(y)", H_y

#this gives the cap_shape column values
cap_shape=list(df_train['f'])

print len(cap_shape)

p=list(np.unique(cap_shape))
print p #this is 6 categories



h_x=0 #entropy 
H_x=[]
#traversing each class label 
for j in range(len(p)):
    h_x=0
    #df1 is datatset containing label p[j], lets say 'b'
    df1=df_train[df_train['f']==p[j]]
    print "df1 is", df1.shape[0]
    #df2 and df3 are datasets containing e and p values respectively having cap_shape='b'
    df2=df1[df1['e']=='e']
    df3=df1[df1['e']=='p']
    #entropy of class p[j[ wrt label e
    c1=float((float(df2.shape[0])/float(len(cap_shape))))
    if c1!=0:
       g1=-1.0*c1*np.log(c1)
    else:
        g1=0.0
    #entropy of class p[j] wrt label p
    c2=float((float(df3.shape[0])/float(len(cap_shape))))
    if c2!=0:
       g2=-1.0*c2*np.log(c2)
    else:
        g2=0.0
    h_x=h_x+g1+g2
    #print h_x
    H_x.append(h_x)
    #print "the entropy of", p[j],"is", h_x
    
entropy=np.sum(H_x)/len(p) 
print "the final entropy of capshape(x1) given y is ", entropy
    

"""