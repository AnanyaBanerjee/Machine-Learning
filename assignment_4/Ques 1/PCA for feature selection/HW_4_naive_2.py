#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:19:36 2018

@author: Mammu
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 12:57:55 2018

@author: Ananya Banerjee

Gaussian Naive Bayes

"""

import numpy as np
from numpy import linalg as lg
import pandas as pd
import math
from cvxopt import matrix, solvers  #this package helps solve quadratic optimization problems
from sklearn import svm, metrics
import operator
import random

#Accuracy for 100 iterations
ACCURACY_final=[]

#average accuracy
AVG_ACC=[]


#dictionary containing all accuracies of k and s pair
K_S=dict()  #iteration number : [k, s, accuracy]

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

#global dictionary containing the mu sigma pairs as values and col number as key
GD=dict()

#function to find index list containing labels 1 and -1
def calc_index_list(labels, df):
    
    count_target_a=0 #number of target for n_(y=1)=col/y=1
    count_target_b=0#number of target for n_(y=-1)=col/y=2
    
    
    #index list containing y=1
    list_of_y_one=[]
    
    #index list containing y=-1
    list_of_y_minus_one=[]
    
    #labels: y=1,-1
    for i in range(len(df)):
        if(labels[i]==1):
            count_target_a+=1
            list_of_y_one.append(i)
        else:
            count_target_b+=1
            list_of_y_minus_one.append(i)
                
    #print "list is", list_of_y_one,"lisy", list_of_y_minus_one , "count", count_target_a, "count b",count_target_b 
    return  list_of_y_one, list_of_y_minus_one , count_target_a, count_target_b 


#function to find mu
def find_mean(df, labels, col, l_y_o, l_y_m_o, count_a,count_b):
    #col contains the column index against whom you want to compute
    
    g=list(df[:,col])
    
    #print "g is",g
    sum_1=0
    sum_minus_1=0
    
    l_y_o=np.sort(l_y_o)
    
    l_y_m_o=np.sort(l_y_m_o)
    
    #print l_y_o, len(l_y_o)
    #print l_y_m_o, len(l_y_m_o)
    #itr=0
    for i in l_y_o:
        c=g[i] #extracts the value of col at index i
        sum_1+=c
        #itr=itr+1
    #print "its is", itr, "count(a)", count_a 
        
    for j in l_y_m_o:
        c=g[j]#extracts the value of col at index i
        sum_minus_1+=c
     
    #print "col", col,"sum1", sum_1,"sum2",sum_minus_1
    mu_1=float(sum_1)/float(count_a)
    mu_2=float(sum_minus_1)/float(count_b)
    return mu_1, mu_2


#function to calculate sigma sig_1 and sig_2
def find_sigma(df, labels, col,l_y_o, l_y_m_o, count_a,count_b ):
    #get mu1 and mu2 corresponding to column col and target 1 and -1
    mu_1, mu_2=find_mean(df, labels, col, l_y_o, l_y_m_o, count_a,count_b)
    #retrive the list of col
    g=list(df[:,col])
    
    sum_1=0
    sum_minus_1=0
    
    for i in l_y_o:
        c=g[i] #extracts the value of col at index i
        c=np.square((c-mu_1))
        sum_1+=c
        
    for j in l_y_m_o:
        c=g[j]#extracts the value of col at index i
        c=np.square((c-mu_2))
        sum_minus_1+=c
           
    sig_1=np.sqrt(float(sum_1)/float(count_a))
    sig_2=np.sqrt(float(sum_minus_1)/float(count_b))
    
    GD[col]=[mu_1, sig_1, mu_2, sig_2]
    
    
#function to perform gaussian
def perform_mle_gaussian(df, labels, count_a, count_b):
    #extracting mu_1,  sig_1,mu_2, sig_2
    sig_1=[]
    mu_1=[]
    mu_2=[]
    sig_2=[]
    for i in GD:
        x=GD[i]
        mu_1.append(x[0])
        sig_1.append(x[1])
        mu_2.append(x[2])
        sig_2.append(x[3])
    
    #print "mu_1", mu_1
   # print "mu_2", mu_2
   # print "sig_1", sig_1
    #print "sig_2", sig_2
    #FOR mu_1 and sig_1
    #for every column
    #list_1: contains probablities that y=1
    list_1=[]
    for i in range(len(df)):
        #print "df", df, df.shape
        row=list(df[i,:]) #contains row
        #print "row len", len(row)
        if(len(row)==1):
            sub=row[0]-mu_1[0]
        else:    
            sub=map(operator.sub, row, mu_1) #subtracts row with mu_1 for each col in row
        sqr=np.power(sub,2) #square(x_i-u_i)    
        sig_sqr=np.power(sig_1,2) #finds sig_1**2 for all elements of sig_1 list  
        sig_mul=np.multiply(sig_sqr,2)
        e1=np.multiply(-1, sqr)
        #print "e1", e1, type(e1),"sig_mul", sig_mul
        if(type(e1)!=list):
            e2=[e1/d for d in sig_mul]
        else:
            e2=[c/d for c, d in zip(e1, sig_mul)]
        ep= np.exp(e2)
        deno=np.multiply((2*math.pi),sig_sqr)
        den_final=[1/z for z in np.sqrt(deno)]
        #den_final=float(1)/float(np.sqrt(deno))
        prob=ep*den_final
        prob_1=np.prod(prob)
        list_1.append(prob_1)
    
    #print "list_1", list_1, len(list_1)
    #FOR mu_2 and sig_2
    #list_2: contains probablities that y=-1
    list_2=[]
    for i in range(len(df)):
        row=list(df[i,:]) #contains row
        if(len(row)==1):
            sub=row[0]-mu_1[0]
        else:    
            sub=map(operator.sub, row, mu_2) #subtracts row with mu_1 for each col in row
        sqr=np.power(sub,2) #square(x_i-u_i)    
        sig_sqr=np.power(sig_2,2) #finds sig_1**2 for all elements of sig_1 list  
        sig_mul=np.multiply(sig_sqr,2)
        e1=np.multiply(-1, sqr)
        if(type(e1)!=list):
            e2=[e1/d for d in sig_mul]
        else:
            e2=[c/d for c, d in zip(e1, sig_mul)]
        ep= np.exp(e2)
        deno=np.multiply((2*math.pi),sig_sqr)
        den_final=[1/z for z in np.sqrt(deno)]
        #den_final=float(1)/float(np.sqrt(deno))
        prob=ep*den_final
        prob_2=np.prod(prob)
        list_2.append(prob_2)
    
    prob_1_t=float(count_a)/float(df.shape[0])
    prob_2_t=float(count_b)/float(df.shape[0])

    list_1=list(np.multiply(list_1, prob_1_t))
    list_2=list(np.multiply(list_2, prob_2_t))

    #print "len is", len(list_1), len(list_2), np.array(list_1).shape
    return list_1, list_2
    
    
#function to perform gaussian
def perform_mle_gaussian_1(df, labels, count_a, count_b, list_of_col):
    #extracting mu_1,  sig_1,mu_2, sig_2
    sig1=[]
    mu1=[]
    mu2=[]
    sig2=[]
    for i in GD:
        x=GD[i]
        mu1.append(x[0])
        sig1.append(x[1])
        mu2.append(x[2])
        sig2.append(x[3])
    
    mu_1=[]
    mu_2=[]
    sig_1=[]
    sig_2=[]
    
    list_of_col=np.sort(list_of_col)

    for g in list_of_col:
        mu_1.append(mu1[g])
        mu_2.append(mu2[g])
        sig_1.append(sig1[g])
        sig_2.append(sig2[g])
   #print "mu_1", mu_1
   # print "mu_2", mu_2
   # print "sig_1", sig_1
    #print "sig_2", sig_2
    #FOR mu_1 and sig_1
    #for every column
    #list_1: contains probablities that y=1
    list_1=[]
    for i in range(len(df)):
        #print "df", df, df.shape
        row=list(df[i,:]) #contains row
        #print "row len", len(row)
        sub=map(operator.sub, row, mu_1) #subtracts row with mu_1 for each col in row
        sqr=np.power(sub,2) #square(x_i-u_i)    
        sig_sqr=np.power(sig_1,2) #finds sig_1**2 for all elements of sig_1 list  
        sig_mul=np.multiply(sig_sqr,2)
        e1=np.multiply(-1, sqr)
        #print "e1", e1, type(e1),"sig_mul", sig_mul
        e2=[c/d for c, d in zip(e1, sig_mul)]
        ep= np.exp(e2)
        deno=np.multiply((2*math.pi),sig_sqr)
        den_final=[1/z for z in np.sqrt(deno)]
        #den_final=float(1)/float(np.sqrt(deno))
        prob=ep*den_final
        prob_1=np.prod(prob)
        list_1.append(prob_1)
    
    #print "list_1", list_1, len(list_1)
    #FOR mu_2 and sig_2
    #list_2: contains probablities that y=-1
    list_2=[]
    for i in range(len(df)):
        row=list(df[i,:]) #contains row
        sub=map(operator.sub, row, mu_2) #subtracts row with mu_1 for each col in row
        sqr=np.power(sub,2) #square(x_i-u_i)    
        sig_sqr=np.power(sig_2,2) #finds sig_1**2 for all elements of sig_1 list  
        sig_mul=np.multiply(sig_sqr,2)
        e1=np.multiply(-1, sqr)
        e2=[c/d for c, d in zip(e1, sig_mul)]
        ep= np.exp(e2)
        deno=np.multiply((2*math.pi),sig_sqr)
        den_final=[1/z for z in np.sqrt(deno)]
        #den_final=float(1)/float(np.sqrt(deno))
        prob=ep*den_final
        prob_2=np.prod(prob)
        list_2.append(prob_2)
    
    prob_1_t=float(count_a)/float(df.shape[0])
    prob_2_t=float(count_b)/float(df.shape[0])

    list_1=list(np.multiply(list_1, prob_1_t))
    list_2=list(np.multiply(list_2, prob_2_t))

    #print "len is", len(list_1), len(list_2), np.array(list_1).shape
    return list_1, list_2
    


    
#function to predict 
def predict_y(list_1, list_2):
    #y_pred contains the predicted values
    y_pred=[]
    for i in range(len(list_1)):
        if list_1[i]>=list_2[i]:
            y_pred.append(1)
        else:
            y_pred.append(-1)
            
    return y_pred


#function to calculate accuracy
def calc_accuracy(y_true,y_pred):
    count=0;
    for i in range(len(y_true)):
        if y_true[i]==y_pred[i]:
            count+=1

    #print count, len(y_true)
    acc=float(count)/float(len(y_true))
    return acc


#function to compute the covariance matrix
def compute_cov_matrix(df):
    W=np.zeros((df.shape[1],df.shape[1])) #(n,p)
    
    for i in range(df.shape[0]):
        if i==60:
            break;
        for j in range(df.shape[0]):
            if j==60:
                break;
            W[i][j]=df[i][j]-np.mean(df[i])
    
    W=np.dot(W.T,W)
    return W


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


#function to calculate pie func
def calculate_pie(chosen_eigen_vec):
    #list of pie
    #print "len of k", len(chosen_eigen_vec)
    #pie_j= 1/k * sum( square(all chosen eigen_vec))
    v_k_1=np.square(chosen_eigen_vec)
    v_k_2=np.sum(v_k_1, axis=0)#%/float(len(chosen_eigen_vec))
    v_k=[float(i)/len(chosen_eigen_vec) for i in v_k_2]
    return v_k
        
################
#next part  

#function to calculate and apply pie distribution
def apply_pi(V, k, Sig):
    chosen_eigen_vec=choose_k_eigen_vec(k, V)
    chosen_eigen_val=choose_k_eigen_val(k, Sig)
    print "chosen eigen vec", chosen_eigen_vec
    #print type(chosen_eigen_vec)
    #sprint chosen_eigen_vec
    print "Chosen Eigen Values", chosen_eigen_val
    #calculate prob distribution
    list_of_pi=calculate_pie(chosen_eigen_vec)
    #now sample s columns independently from pi
    #print list_of_pi
    return list_of_pi
    
#function to sample data
def sample_data(list_of_pie, s):
    #list of columns
    col_num=[]
    #s columns are sampled randomly
    for i in range(s):
        c=np.random.choice(list_of_pie)
        ind=list_of_pie.index(c)
        col_num.append(ind)
    
    return col_num

#function to sample data
def sample_data_1(list_of_pie, s):
    #print "list of pie", len(list_of_pie)
    #list of columns
    col_num=[]
    #list of pie_w
    list_of_pie_w=[]
    sum1=0
    j=0
    for i in range(len(list_of_pie)):
        if i==0:
            list_of_pie_w.append(sum1)
        else:
            sum1+=list_of_pie[j]
            j=j+1
            list_of_pie_w.append(sum1)
            
    #print "list of piw_w", list_of_pie_w, len(list_of_pie_w)
    #exit()
    #s columns are sampled randomly
    for i in range(s):
        #generate a random number
        x=random.random()
        for j in range(len(list_of_pie_w)):
            if list_of_pie_w[j]>x:
                c=j-1
                break;
        ind=list_of_pie_w.index(list_of_pie_w[c])
        #print "index is", ind
        col_num.append(ind)
    
    return col_num


#function to perform naive bayes
def naive_bayes(df, labels, k, list_of_col):
    
    #normalize the data
    mean=np.mean(df, axis=0)
    sig=np.std(df, axis=0)
    df_normalized=(df-mean)/sig
    
    #find list of indexes
    list_of_y_one, list_of_y_minus_one , count_a, count_b  = calc_index_list(labels, df_normalized)
    
    #find mu and sigma
    for col in range(df.shape[1]):
        if(col==60):
            break;
        find_sigma(df_normalized, labels, col,list_of_y_one, list_of_y_minus_one, count_a,count_b )

    #list_1 contains the probablity that y=1 is predicted
    #list_2 contains the probability that y=-1 is predcited
    list_1, list_2=perform_mle_gaussian_1(df, labels, count_a, count_b, list_of_col)

    #prob_1=float(count_a)/float(df.shape[0])
    #prob_2=float(count_b)/float(df.shape[0])
    #print "list_2", list_2
    #list_1=list(np.multiply(list_1, prob_1))
    #list_2=list(np.multiply(list_2, prob_2))
    
    #print "list_1", list_1
    #print "list_2", list_2
    y_pred=predict_y(list_1, list_2)
    
    Accuracy=calc_accuracy(labels,y_pred)
    print "Accuracy  is", Accuracy
    return Accuracy

def test(df_test, y_test):
    #testing the test set
    count_a_test=list(y_test).count(1)
    count_b_test=list(y_test).count(-1)

    #list_1 contains the probablity that y=1 is predicted
    #list_2 contains the probability that y=-1 is predcited
    list_1_t, list_2_t=perform_mle_gaussian(df_test, y_test, count_a_test, count_b_test)

    y_pred_test=predict_y(list_1_t, list_2_t)

    Accuracy=calc_accuracy(y_test,y_pred_test)
    print "Accuracy for test is", Accuracy
    return Accuracy

#function to perform prob1: second part third
def try_func(df,labels):
    k=[1,2,3,4,5,6,7,8,9,10]
    s=range(1,20)
    itr=0
    for i in k:
        for j in s:
            avg=0
            #final_aVg=0
            cov_mat=compute_cov_matrix(df)
            U, Sig, V=perform_svd(cov_mat)
            
            list_of_pi=apply_pi(V, i, Sig)
            #print "Checking if pie is okay", np.sum(list_of_pi)
    
            for b in range(100):
                col_num=sample_data_1(list_of_pi, j)
                #print "columns chosen are", col_num
                
                #extract these columns from the dataset
                data=df[:,col_num]
                #print "data", data.shape
                acc=naive_bayes(data, labels, i, col_num)
                avg+=acc
                #acc=test(data, labels)
                ACCURACY_final.append(acc)
                K_S[itr]=[i, j, acc]
                itr=itr+1
            
            
            #final_aVg=float(avg)/100
            AVG_ACC.append(float(avg)/float(100))
            
    
    
"""
#part 1
"""
k=6
list_of_col=range(60)
naive_bayes(df, labels, k, list_of_col)

cov_mat=compute_cov_matrix(df)
#print "cov_mat", cov_mat
U, Sig, V=perform_svd(cov_mat)
chosen_eigen_vec=choose_k_eigen_vec(k, V)
chosen_eigen_val=choose_k_eigen_val(k, Sig)

"""
part 2-3
"""
#test(df_test, y_test)
try_func(df_test,y_test)

ma=np.max(ACCURACY_final)
it=ACCURACY_final.index(ma)
best=K_S[it]
print "Best value of k and s", best[0], best[1], "and accuracy is", best[2]






    
    
    
    
    
    
    
    