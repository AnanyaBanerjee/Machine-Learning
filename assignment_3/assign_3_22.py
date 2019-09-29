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

#list of errors
errors=[]

#list of hypothesis all
hypo=[]

#list of chosen hypothesis
chosen_hp=[]

#list of alphas
alpha_f=[]

#list of accuracies_train
accuracy_train=[]

#dictionary containing attributes and corresponding splits
split=dict()



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
    #accuracy_train.append(accuracy)
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
    split[err]=[1, li_of_hp, attr1]
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
    split[err]=[2, li_of_hp, attr1]
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
        errors.append(err1)
        errors.append(err2)
        errors.append(err3)
        errors.append(err4)
        hypo.append(pred1)
        hypo.append(pred2)
        hypo.append(pred3)
        hypo.append(pred4)

        Hypothesis[x+str(1)]=[err1,pred1]
        Hypothesis[x+str(2)]=[err2,pred2]
        Hypothesis[x+str(3)]=[err3,pred3]
        Hypothesis[x+str(4)]=[err4,pred4]
        
        
#function to find alpha
def compute_alpha(y, y_pred, alpha_i, t):
    #
    sum_n=0.0 #sum numerator
    sum_d=0.0 #sum denominator
    sum_o=0.0 #this sum -> -v
    #find sum of all alpha_i * h_t for t
    # alpha_i.append(-1.0)
    # y_pred.append(-1.0)
    for i in range(len(list(Hypothesis.keys()))-1):
        if i==80:
           break;
        if i!=t:
           sum_o=sum_o+np.sum(np.multiply(alpha_i[i], y_pred[i]))
    
    
    for i in range(len(y)):
        if y[i]==y_pred[i]:
            sum_n=sum_n+np.exp(-y[i]*sum_o)
        else:
            sum_d=sum_d+np.exp(-y[i]*sum_o)
        
    final_alpha=0.5*math.log(float(sum_n)/float(sum_d))
    alpha_f.append(final_alpha)
    return final_alpha

#function to perform coordinate descent
def coordinate_descent(df, max_iter):
    alpha_i=[float(1)/float(df.shape[1]) for i in range(len(df[target]))]
    for i in range(max_iter):
        t=random.choice(errors)
        ind=errors.index(t)
        y_pred=hypo[ind]
        chosen_hp.append(y_pred)
        #print "y is", y_pred
        alpha_updated=compute_alpha(df[target], y_pred, alpha_i, t)
        H[alpha_updated]=y_pred
        ac=find_accuracy(df[target], y_pred)
        accuracy_train.append(ac)
    
    
    #print "Final alpha is", alpha_updated
    #return alpha_updated

#function to perform coordinate descent
def coordinate_descent_1(df, max_iter):
    alpha_i=[float(1)/float(df.shape[1]) for i in range(len(df[target]))]
    for i in range(max_iter):
        t=random.choice(errors)
        ind=errors.index(t)
        y_pred=hypo[ind]
        chosen_hp.append(y_pred)
        #print "y is", y_pred
        alpha_updated=compute_alpha(df[target], y_pred, alpha_i, t)
        H[alpha_updated]=y_pred
        ac=find_accuracy(df[target], y_pred)
        accuracy_train.append(ac)
    
    
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
   
    #print list_of_col
    
    #making a dataframe
    df=pd.DataFrame(dataset_train, columns=list_of_col)
    df=df.fillna(0)

    #print df.shape
    for i in range(len(df[target])):
        if df[target].loc[i]==0:
            df[target].loc[i]=-1
    
      
    #df[target].loc[80]=-1.0
    df= df.drop(df.index[80])
    target_col.append(df[target])
    #print df
    
    #df_i=df
    #del df_i[target]
    find_hypothesis(df)
    #AdaBoost(df, 1, list_of_col)
    coordinate_descent(df, 1000)
    #calculate the loss function
    lo=0.0
    #j=0
    loss=0.0
    print H.keys()
    #print H.values()
    for j in range(df.shape[1]):
        for i in list(H.keys()):
            pred=H[i]
            lo+=np.sum(np.multiply(i, pred[j]))
        loss+=np.exp(-df[target].loc[j]*lo)
    
    print "the Loss is", loss

    for b in range(len(hypo)):
        acc=find_accuracy(df[target], hypo[b])
        print "accuracy for alpha", alpha_f[b], "is", acc
        #accuracy_train.append(acc)
    
    """
    sum1=[0 for i in range(len(chosen_hp))]
    j=0
    for i in range(len(chosen_hp)):
        #print hypo[i], alpha_f[i]
        c=chosen_hp[i] #contains the hypothesis column
        print "c", c, alpha_f[j]
        b=[c[i]*alpha_f[j] for i in range(len(c))]
        print "b is", b
        sum1=[x + y for x, y in zip(b, sum1)]
        print "sum1: ", sum1
        j=j+1
        
    #print sum1
    final_h=[]
    for x in sum1:
        if(np.sign(x)):
            final_h.append(1.0)
        else:
            final_h.append(-1.0)
            
    acc=find_accuracy(df[target], final_h)
    print "Accuracy on training set is", acc
    #print "len of hypo", len(hypo)   , len(errors) 
    #print "alpha is", alpha_f, len(alpha_f), "chosen h is", len(chosen_hp)
    #print "len of acc", len(accuracy_train)

    #test set
    #making a dataframe
    df_test=pd.DataFrame(dataset_test, columns=list_of_col)
    df_test=df.fillna(0)

    #print df.shape
    for i in range(len(df_test[target])):
        if df_test[target].loc[i]==0.0:
            df_test[target].loc[i]=-1.0
    
    cc=find_accuracy(df_test[target], final_h)
    print "Accuracy on test set is", cc

    #it
    maxi=np.max(accuracy_train)
    inde=accuracy_train.index(maxi)
    h_chosen=chosen_hp[inde]
    cc=find_accuracy(df[target], h_chosen)
    print "Accuracy on train set using chosen h is", cc
    cc2=find_accuracy(df_test[target], h_chosen)
    print "Accuracy on test set using chosen h is", cc2
    """
    maxi=np.max(alpha_f) #choose the alpha with the most trust
    print "alpha chosen is",maxi
    ind=alpha_f.index(maxi) #index of chosen aplha
    final_h=chosen_hp[ind]
    #test set
    #making a dataframe
    df_test=pd.DataFrame(dataset_test, columns=list_of_col)
    df_test=df_test.fillna(0)

    #print df.shape
    for i in range(len(df_test[target])):
        if df_test[target].loc[i]==0.0:
            df_test[target].loc[i]=-1.0
    

    cc=find_accuracy(df[target], final_h)
    print "Accuracy on train set using chosen h is", cc
    #print "df_test[target]", len(df_test[target]), len(alpha_f)
    

MAIN()
