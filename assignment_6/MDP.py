#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 13:31:26 2018

@author: Ananya Banerjee

Markov Decision Process (MDP)
"""

import pandas as pd
import numpy as np
from itertools import islice

#reward R(s,a): SXA -> R matrix
R=[0.0, 1, 0.5, 0.5, 0.5, 0.0, 1, 0.5, -1, 0.5, 0.0, 0.5, -1, 0.5, 0.5, 0.0]
R=np.array(R).reshape(4,4)

#transition T(s,a)
T=[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]
T=np.array(T).reshape(4,4)

#State list: S1==> 1, S2==> 2, S3==> 3, S4==> 4
S=[1, 2, 3, 4]

#Action list: A1=>1, A2==>2, A3==>3, A4==>4
A=[1, 2, 3, 4]

#function to get reward from reward matrix
def fetch_reward(i, j):
    #i defines state
    #j defines action
    return R[i][j]

#function to check if two lists are equal:
def check_equal(V_t, V_t_1):
    equal=0
    for i in range(len(V_t)):
        if(V_t[i]==V_t_1[i]):
                equal+=1
                
    if equal==len(V_t):
        return 1 #1 implies both are equal
    else:
        return 0 #0 implies both are not equal
        
#function to return value value
def calculate_value(s, a, gamma, V):
    reward=R[s-1][a] #gives the reward when we go from state s and take action a
    #get value of 
    ind=T[s-1][a] #get value of T(s, a) wjere a=pi(s)
    #print T[s-1][a], "s is", s,"a is", a 
    V_pi=V[ind-1]  #get V(T(s,a))
    val= reward+gamma*V_pi #perform  R(s, a)+gamma*V(T(s,a))
    return val
    
        
#function to find V_pi(s) for every state s
def find_value_at_s(s, gamma, V):
    #list of values for state s 
    List_of_val=[]
    #for all actions : A=[1,2,3,4]
    for i in range(len(A)):
        val=calculate_value(s, i, gamma, V)
        List_of_val.append(val)
        
    return List_of_val
        
    


#function to perform value iteration
def Value_Iteration(gamma):
    #calculating Value function using Value Iteration
    #V=[0,0,0,0]
    V=np.random.random_sample(4)
    #V_t=[1,1,1,1]
    V_t=np.random.random_sample(4)
    #array of policy
    pi=[0,0,0,0]
    
    while(check_equal(V_t, V)!=1):
        #for each state calulate V_t_1
        V_t=np.copy(V)
        #for s=1: 
        V_s_1=find_value_at_s(1, gamma, V_t)
        print "V_S_1", V_s_1
        V[0]=np.max(V_s_1)
        pi[0]=V_s_1.index(V[0])
        
        #for s=2:
        V_s_2=find_value_at_s(2, gamma, V_t)
        V[1]=np.max(V_s_2)
        print "V_S_2", V_s_2
        pi[1]=V_s_2.index(V[1])
        
        #for s=3: 
        V_s_3=find_value_at_s(3, gamma, V_t)
        V[2]=np.max(V_s_3)
        print "V_S_3", V_s_3
        pi[2]=V_s_3.index(V[2])
        
        #for s=4: 
        V_s_4=find_value_at_s(4, gamma, V_t)
        V[3]=np.max(V_s_4)
        print "V_S_4", V_s_4
        pi[3]=V_s_4.index(V[3])
        #V contains V_t_plus_1 for this iteration
        print "Updated V is", V, "and V_t is", V_t
    
    print "The final value after value iteration is" ,V
    print "The optimum policy is", pi
    
            
Value_Iteration(0.8)   

#   
        
        
        
        
    
