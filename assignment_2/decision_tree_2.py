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



#fucntion to apply tree_1: 
"""
    attr1
     - -
   -    -
 0        attr2
        -    - 
     -         -
    0            attr3
                -     -
              -        -
            0            1
"""
def apply_tree1(attr1, attr2, attr3, df, itr, list_of_col,w):
    print "Apply tree_1"
    g=np.float64(0)  #0.0
    h=np.float64(1)  #1.0
    df1_1=df[df[attr1]==g] #attr1=0 #give label 0
    df1_2=df[df[attr1]==h] #attr1=1
    #split the dataset using attr2
    df2_1=df1_2[df1_2[attr2]==g] #attr1=0, attr2=0 #give label 0
    df2_2=df1_2[df1_2[attr2]==h] #attr1=0, attr2=1
    #split the dataset using attr3
    df3_1=df2_2[df2_2[attr3]==g] #attr1=0, attr2=0, attr3=0 #give label 0
    df3_2=df2_2[df2_2[attr3]==h] #attr1=0, attr2=0, attr3=1 #give label 1
    
    #000, 001, 010, 011, 100, 101, 110,111
    prediction=[0,0,0,0,0,0,0,1]
    
    #list of split datasets
    list_of_split_datasets=[df1_1, df2_1, df3_1, df3_2]
    #dictionary containing split datasets where key is iteration number
    ab=dict()
    
    ab[itr]=list_of_split_datasets
    
    #add this to global datasets
    GD[itr]=ab
    
    #
    df[str(target)+'pred']=[0.0 for i in range(len(df[target]))]
    
    for i in range(len(df[target])):
        if(df[attr1].loc[i]=='0.0'):
           df[str(target)+'pred'].loc[i]=0.0
        else:
             if(df[attr2].loc[i]=='0.0'):
                df[str(target)+'pred'].loc[i]=0.0
             else:
                  if(df[attr3].loc[i]=='0.0'):
                     df[str(target)+'pred'].loc[i]=0.0
                  else:
                      df[str(target)+'pred'].loc[i]=1.0
    pred=df[str(target)+'pred']
    PRED[str(itr)+'1']=list(pred)
    acc=find_accuracy(df[target], df[str(target)+'pred'])
    err=calc_error(w, df[str(target)+'pred'], df[target])
    print "accuracy is", acc
    print "error is", err 
    return err, pred

#fucntion to apply tree_2: 
"""
                  attr1
                   - -
                 -    -
           attr2        1
         -    - 
      -         -
    attr3         1
   -     -
  -       -
 0          1 
"""
def apply_tree2(attr1, attr2, attr3, df, itr, list_of_col,w):
    print "Apply tree_2"
    g=np.float64(0)  #0.0
    h=np.float64(1)  #1.0
    df1_1=df[df[attr1]==g] #attr1=0 
    df1_2=df[df[attr1]==h] #attr1=1 #give label 1
    #split the dataset using attr2
    df2_1=df1_1[df1_1[attr2]==g] #attr1=0, attr2=0 
    df2_2=df1_1[df1_1[attr2]==h] #attr1=0, attr2=1 #give label 1
    #split the dataset using attr3
    df3_1=df2_1[df2_1[attr3]==g] #attr1=0, attr2=0, attr3=0 #give label 0
    df3_2=df2_1[df2_1[attr3]==h] #attr1=0, attr2=0, attr3=1 #give label 1
    
    #000, 001, 010, 011, 100, 101, 110,111
    prediction=[1,1,1,1,1,1,1,0]
    
    #list of split datasets
    list_of_split_datasets=[df1_2, df2_2, df3_1, df3_2]
    #dictionary containing split datasets where key is iteration number
    ab=dict()
    
    ab[itr]=list_of_split_datasets
    
    #add this to global datasets
    GD[itr]=ab
    
    #
    df[str(target)+'pred']=[0.0 for i in range(len(df[target]))]
    
    for i in range(len(df[target])):
        if(df[attr1].loc[i]=='1.0'):
           df[str(target)+'pred'].loc[i]=1.0
        else:
             if(df[attr2].loc[i]=='1.0'):
                df[str(target)+'pred'].loc[i]=1.0
             else:
                  if(df[attr3].loc[i]=='1.0'):
                     df[str(target)+'pred'].loc[i]=1.0
                  else:
                      df[str(target)+'pred'].loc[i]=0.0
    pred=df[str(target)+'pred']
    PRED[str(itr)+'2']=list(pred)
    acc=find_accuracy(df[target], df[str(target)+'pred'])
    err=calc_error(w, df[str(target)+'pred'], df[target])
    print "accuracy is", acc
    print "error is", err 
    return err, pred

#fucntion to apply tree_4: 
"""
                  attr1
                   - -
                 -    -
           attr2        1
         -    - 
      -         -
     0         attr3         
               -     -
              -       -
             0          1 
"""
def apply_tree4(attr1, attr2, attr3, df, itr, list_of_col,w):
    print "Apply tree_4"
    g=np.float64(0)  #0.0
    h=np.float64(1)  #1.0
    df1_1=df[df[attr1]==g] #attr1=0 
    df1_2=df[df[attr1]==h] #attr1=1 #give label 1
    #split the dataset using attr2
    df2_1=df1_1[df1_1[attr2]==g] #attr1=0, attr2=0 #give label 0
    df2_2=df1_1[df1_1[attr2]==h] #attr1=0, attr2=1 
    #split the dataset using attr3
    df3_1=df2_2[df2_2[attr3]==g] #attr1=0, attr2=1, attr3=0 #give label 0
    df3_2=df2_2[df2_2[attr3]==h] #attr1=0, attr2=1, attr3=1 #give label 1
    
    #000, 001, 010, 011, 100, 101, 110,111
    prediction=[0,0,0,1,1,1,1,1]
    
    #list of split datasets
    list_of_split_datasets=[df1_2, df2_1, df3_1, df3_2]
    #dictionary containing split datasets where key is iteration number
    ab=dict()
    
    ab[itr]=list_of_split_datasets
    
    #add this to global datasets
    GD[itr]=ab
    
    #
    df[str(target)+'pred']=[0.0 for i in range(len(df[target]))]
    
    for i in range(len(df[target])):
        if(df[attr1].loc[i]=='1.0'):
           df[str(target)+'pred'].loc[i]=1.0
        else:
             if(df[attr2].loc[i]=='1.0'):
                df[str(target)+'pred'].loc[i]=1.0
             else:
                  if(df[attr3].loc[i]=='1.0'):
                     df[str(target)+'pred'].loc[i]=0.0
                  else:
                      df[str(target)+'pred'].loc[i]=0.0
    pred=df[str(target)+'pred']
    PRED[str(itr)+'4']=list(pred)
    acc=find_accuracy(df[target], df[str(target)+'pred'])
    err=calc_error(w, df[str(target)+'pred'], df[target])
    print "accuracy is", acc
    print "error is", err 
    return err, pred

#fucntion to apply tree_3: 
"""
                  attr1
                   - -
                 -    -
                0      attr2        
                      -    - 
                    -         -
               attr3           1
               -     -
              -       -
             0          1 
"""
def apply_tree3(attr1, attr2, attr3, df, itr, list_of_col,w):
    print "Apply tree_3"
    g=np.float64(0)  #0.0
    h=np.float64(1)  #1.0
    df1_1=df[df[attr1]==g] #attr1=0 #give label 0
    df1_2=df[df[attr1]==h] #attr1=1 
    #split the dataset using attr2
    df2_1=df1_2[df1_2[attr2]==g] #attr1=1, attr2=0 
    df2_2=df1_2[df1_2[attr2]==h] #attr1=1, attr2=1 #give label 1
    #split the dataset using attr3
    df3_1=df2_1[df2_1[attr3]==g] #attr1=1, attr2=0, attr3=0 #give label 0
    df3_2=df2_1[df2_1[attr3]==h] #attr1=1, attr2=0, attr3=1 #give label 1
    
    #000, 001, 010, 011, 100, 101, 110,111
    prediction=[0,0,0,0,0,1,1,1]
    
    #list of split datasets
    list_of_split_datasets=[df1_1, df2_2, df3_1, df3_2]
    #dictionary containing split datasets where key is iteration number
    ab=dict()
    
    ab[itr]=list_of_split_datasets
    
    #add this to global datasets
    GD[itr]=ab
    
    #
    df[str(target)+'pred']=[0.0 for i in range(len(df[target]))]
    
    for i in range(len(df[target])):
        if(df[attr1].loc[i]=='0.0'):
           df[str(target)+'pred'].loc[i]=0.0
        else:
             if(df[attr2].loc[i]=='1.0'):
                df[str(target)+'pred'].loc[i]=1.0
             else:
                  if(df[attr3].loc[i]=='1.0'):
                     df[str(target)+'pred'].loc[i]=1.0
                  else:
                      df[str(target)+'pred'].loc[i]=0.0
    pred=df[str(target)+'pred']
    PRED[str(itr)+'3']=list(pred)
    acc=find_accuracy(df[target], df[str(target)+'pred'])
    err=calc_error(w, df[str(target)+'pred'], df[target])
    print "accuracy is", acc
    print "error is", err 
    return err, pred

#fucntion to apply tree_5: 
"""
                  attr2
                  -   -
                -      -
             attr1      attr3     
            -    -     -    - 
          -       -   -       -
          0       1   0        1
             
"""
def apply_tree5(attr1, attr2, attr3, df, itr, list_of_col,w):
    print "Apply tree_5"
    g=np.float64(0)  #0.0
    h=np.float64(1)  #1.0
    df1_1=df[df[attr2]==g] #attr2=0 
    df1_2=df[df[attr2]==h] #attr2=1 
    #split the dataset using attr1
    df2_1=df1_1[df1_1[attr1]==g] #attr2=0, attr1=0 #label : 0 
    df2_2=df1_1[df1_1[attr1]==h] #attr2=0, attr1=1 #label : 1
    #split the dataset using attr3
    df3_1=df2_2[df2_2[attr3]==g] #attr2=1,  attr3=0 #give label 0
    df3_2=df2_2[df2_2[attr3]==h] #attr2=1,  attr3=1 #give label 1
    
    #000, 001, 010, 011, 100, 101, 110,111
    prediction=[0,0,0,1,1,1,0,1]
    
    #list of split datasets
    list_of_split_datasets=[df1_1, df2_2, df3_1, df3_2]
    #dictionary containing split datasets where key is iteration number
    ab=dict()
    
    ab[itr]=list_of_split_datasets
    
    #add this to global datasets
    GD[itr]=ab
    
    #
    df[str(target)+'pred']=[0.0 for i in range(len(df[target]))]
    
    for i in range(len(df[target])):
        if(df[attr2].loc[i]=='0.0'):
            if(df[attr1].loc[i]=='0.0'):
               df[str(target)+'pred'].loc[i]=0.0
            else:
               df[str(target)+'pred'].loc[i]=1.0
        else:
             if(df[attr3].loc[i]=='0.0'):
                df[str(target)+'pred'].loc[i]=0.0
             else:
                df[str(target)+'pred'].loc[i]=1.0
    pred=df[str(target)+'pred']
    PRED[str(itr)+'5']=list(pred)
    acc=find_accuracy(df[target], df[str(target)+'pred'])
    err=calc_error(w, df[str(target)+'pred'], df[target])
    print "accuracy is", acc
    print "error is", err 
    return err, pred

#function to find hypothesis space
def find_hypothesis_space(df, max_iter,list_of_col, w):
    
    for itr in range(len(list(df.columns))):
        l=list(df.columns.to_series().sample(3))
        print "chosen attributes are:", l
        err1, pred1=apply_tree1(l[0], l[1], l[2], df, itr, list_of_col,w)
        print err1
        err2, pred2=apply_tree2(l[0], l[1], l[2], df, itr, list_of_col,w)
        print err2
        err3,pred3=apply_tree3(l[0], l[1], l[2], df, itr, list_of_col,w)
        print err3
        err4, pred4=apply_tree4(l[0], l[1], l[2], df, itr, list_of_col,w)
        print err4
        err5, pred5=apply_tree5(l[0], l[1], l[2], df, itr, list_of_col,w)
        print err5
        #The CHOICE dictionary has err as key
        #The CHOICE has tree applied as key,  pred and attribute it split on as keys
        CHOICE[err1]=[1, pred1, l]
        CHOICE[err2]=[2, pred2, l]
        CHOICE[err3]=[3, pred3, l]
        CHOICE[err4]=[4, pred4, l]
        CHOICE[err5]=[5, pred5, l]
        
        #CHOICE[str(l)]=[err1, err2, err3, err4, err5]
        #hypothesis dict 
        Hypothesis[err1]=pred1
        Hypothesis[err2]=pred2
        Hypothesis[err3]=pred3
        Hypothesis[err4]=pred4
        Hypothesis[err5]=pred5



#function to update hypothesis
def update_hypothesis(mini, updated_w, df, itr):
    #first find which combination of attributes generated this hypothesis with err mini
    li=CHOICE[mini]
    #li contains tree number, pred col and list of attribute
    apply_tree=li[0]
    pred_col=li[1]
    list_of_attr=li[2]
    #fit the model with your chosen combinations
    if(apply_tree==1):
       err, pred=apply_tree1(list_of_attr[0], list_of_attr[1], list_of_attr[2], df, itr, list_of_attr,updated_w)
       print err
    elif(apply_tree==2):
       err, pred=apply_tree2(list_of_attr[0], list_of_attr[1], list_of_attr[2], df, itr, list_of_attr,updated_w)
       print err
    elif(apply_tree==3):
       err, pred=apply_tree3(list_of_attr[0], list_of_attr[1], list_of_attr[2], df, itr, list_of_attr,updated_w)
       print err
    elif(apply_tree==4):
       err, pred=apply_tree4(list_of_attr[0], list_of_attr[1], list_of_attr[2], df, itr, list_of_attr,updated_w)
       print err
    else:
       err, pred=apply_tree5(list_of_attr[0], list_of_attr[1], list_of_attr[2], df, itr, list_of_attr,updated_w)
       print err
       
    #update hypothesis in dict Hypothesis
    Hypothesis[err]=pred
    return err, pred
    

#function to perform adaboost 
def AdaBoost(df, max_iter,list_of_col):
    M=len(df[target])
    #initialize weights 
    df[target].loc[80]=0.0
    w=[float(1/float(M)) for i in range(len(df[target]))]
    #hypothesis space building 
    find_hypothesis_space(df, max_iter,list_of_col, w)
        
    for itr in range(max_iter):
        #choose the hypothesis with the minimum error
        mini=np.min(list(Hypothesis.keys()))
        print "minimum is", mini
        #find alpha_t which indicates how much value you give to your hypothesis
        ert=float(1-mini)/float(mini)
        alpha_t=0.5*math.log(ert)
        #chosen h is:
        h=Hypothesis[mini]
        #update the weights for the chosen hypothesis
        w=update_weights(h, mini, alpha_t, w, itr)
        #delete the hypothesis from dictionary
        Hypothesis.pop(mini)
        #update the hypothesis and add it to dictionry H
        err,pred=update_hypothesis(mini, w, df, itr)
        #update the dictionary H as 
        H[alpha]=[err, pred]
    
    

  
    
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
    target_col.append(df[target])
    
    #df_i=df
    #del df_i[target]
   
    AdaBoost(df, 1, list_of_col)
    """
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
    """
    
MAIN()