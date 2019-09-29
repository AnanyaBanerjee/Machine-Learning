#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 21:24:53 2018

@author: Ananya Banerjee

Create NN Dataset
"""

import numpy as np
import pandas as pd
import random
from random import *


#function to find number of 1's in an binary number
def find_ones(number):
    #print "list", number, type(number)
    #l converts 1110011 ==> "1110011"
    #l=str(number)
    #now this converts to list : "1110011"==> ['1','1','1','0','0','1','1']
    #li=list(number)
    #now simply count the ones in list l1 and return number of 1's in binary representation of a number
    number_of_ones=number.count(1)
    #print "The number is", number_of_ones
    
    return number_of_ones


#function to take input
def take_input(size):
    #data input list
    data_input=[]

    #take 10 binary inputs
    for i in range(size):
        #take input
        #j=int(raw_input("Enter your binary input"),2)
        #j=raw_input("Enter you binary input")
        #j1=format(int(j, 2), '{fill}{width}b'.format(width=digits, fill=0))
        randBinList = lambda n: [randint(0,1) for b in range(1,n+1)]
        j1=randBinList(10)
        #print "your input is", j1
        #j=input("Enter your binary input")
        data_input.append(j1)
   
    return data_input


#function to calculate output
def find_output(data_input):
    #data output list 
    data_output=[]
    for i in data_input:
        number_of_ones= find_ones(i)
        #print "number of ones", number_of_ones
        if number_of_ones%4==0:
            data_output.append(1)
        else:
            data_output.append(0)
    return data_output


def MAIN():
    #how many digits does the binary number have
    #digits=int(input("Enter how many digit binary number do you want to enter"))
    #take input
    data_input=take_input(100)   
    #generate output
    data_output=find_output(data_input)
    #print input and output
    #print "Input is", data_input
    #print "Output is",data_output
    #construct columns
    list_of_col=["x"+str(i) for i in range(1,10+1)]
    list_of_col.append("y") #constructing target column
    #construct a dataset
    df=pd.DataFrame(columns=list_of_col)
    j=0
    print "columns are", list_of_col
    #put data in it
    for i in data_input:
        #row=list(i)
        i.append(data_output[j])
        df.loc[j]=i
        j=j+1
    
    #print "your data frame is", df
    file_name="data_set_generated.csv"
    df.to_csv(file_name, encoding='utf-8', index=False)
    

MAIN()


