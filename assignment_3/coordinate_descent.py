#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 20:22:32 2018

@author: Luxyfufu
"""
import math
import random
header = ["OVERALL_DIAGNOSIS", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22"]

TEST_LABEL = header.index("OVERALL_DIAGNOSIS")
ATTR = 2
LABEL = header[0]

def input():

	Input_Vector = []

	with open("heart_train.data", "r") as filestream:
		for line in filestream:
			currentline = line.strip()
			currentline = line.split(",")

			Input_Element = [None]*23
			for i in range(len(header)):
				if(i == TEST_LABEL):
					#print(currentline[i], "AND 0 ARE EQUAL ", int(currentline[i]) == 0)
					if(int(currentline[i]) == 0):
						Input_Element[i] = -1
						#print(Input_Element[i])
					else:
						Input_Element[i] = int(currentline[i])
				else:
					Input_Element[i] = int(currentline[i].rstrip('\r\n'))

			Input_Vector.append(Input_Element)


	return Input_Vector;

def test_input():
    Test_Vector = []

    with open("heart_test.data", "r") as filestream:
        for line in filestream:
            currentline = line.strip()
            currentline = line.split(",")

            Test_Element = [None]*23
            for i in range(len(header)):
                if(i == TEST_LABEL):
                    if(int(currentline[i]) == 0):
                        Test_Element[i] = -1
                    else:
                        Test_Element[i] = int(currentline[i])
                else:
                    Test_Element[i] = int(currentline[i])
                    
            Test_Vector.append(Test_Element)


	return Test_Vector;

# FIRST, MAKE THE TREES
def test():
    Input_Vector = []
    Input_Vector = input()
    
    Test_Vector = []
    Test_Vector = test_input()
    # Here, the trees are just going to be arrays of the form [a, b, attribute], 
    # where a is the label for left split and b is label for right split
    TREES = make_trees()
    ALPHA_LIST = make_alphas(len(TREES))
    
        
    ALPHA_LIST = coordinate_descent(TREES, ALPHA_LIST, Input_Vector)

    print(max(ALPHA_LIST), "at: ", ALPHA_LIST.index(max(ALPHA_LIST)))
    
    # BEST HYPOTHESIS SPACE IS MAX ALPHA
    BEST_SPACE = ALPHA_LIST.index(max(ALPHA_LIST))
    acc = accuracy(TREES[BEST_SPACE], Test_Vector)
    print("ACCURACY: ", acc)
    
    LOSS = calculate_loss(TREES, ALPHA_LIST, Input_Vector)
    print(LOSS)
    
    
    ALPHA_LIST = make_alphas(len(TREES))
    
    W = [1.0 / len(Input_Vector)] * len(Input_Vector)
    for i in range(20):
        print("At iteration: ", i)
        W = adaboost(TREES, Input_Vector, W, Test_Vector)
    
    
    
    #print(len(TREES))
    #print(ALPHA_LIST)

def adaboost(TREES, Input_Vector, W, Test_Vector):
    # UPDATE LIST OF ERRORS
    ERROR_LIST = []
    ALPHA_LIST = []
    
    
    for t in range(len(TREES)):
        ErrorSum = 0
        for m in range(len(Input_Vector)):
            ht = return_label(TREES[t], Input_Vector[m])
            y = Input_Vector[m][TEST_LABEL]
            if ht != y:
                ErrorSum += W[m]
        ERROR_LIST.append(ErrorSum)
    
    # NOW THAT WE HAVE OUR ERROR LIST, CALCULATE ALPHAS
    for a in range(len(ERROR_LIST)):
        a = .5 * math.log1p((1.0 - ERROR_LIST[a])/ERROR_LIST[a])
        ALPHA_LIST.append(a)
        
    minError = min(ERROR_LIST)
    minErrorIndex = ERROR_LIST.index(minError)
    
    #print("MIN ERROR: ", minError)
    # NOW THAT WE HAVE ALPHAS, LETS UPDATE THEM WEIGHTS
    for m in range(len(W)):
        ht = return_label(TREES[minErrorIndex], Input_Vector[m])
        y = Input_Vector[m][TEST_LABEL]
        a = ALPHA_LIST[minErrorIndex]
        
        
        if(minError < 1):
            W[m] = (W[m] * math.exp(-1 * y * ht * a)) / (2 * math.sqrt(minError * (1 - minError)))
        elif(minError > 1):
            minError -= 1
            W[m] = (W[m] * math.exp(-1 * y * ht * a)) / (2 * math.sqrt( minError * (1 - minError)))
    
    
    BEST_TREE = TREES[minErrorIndex]
    print("Accuracy in Adaboost for Test Data is: ", accuracy(BEST_TREE, Test_Vector))
    print("ALPHA IS: ", ALPHA_LIST[minErrorIndex])
    
    return W
        
    
    

def coordinate_descent(TREES, ALPHA_LIST, Input_Vector):
    for WEW in range(1000):
        # FIRST, PICK RANDOM HYPOTHESIS
        TREES_INDEX = random.randint(0, len(TREES)-1)
        # THEN: CALCULATE ALPHA T
        # IN alphat*ht(xm)
        topsum = 0
        bottomsum = 0
        for m in range(len(Input_Vector)):    
            SUM_ALL_T = 0
            for t in range(len(TREES)):
                if t!= TREES_INDEX:
                    ht = return_label(TREES[t], Input_Vector[m])
                    a = ALPHA_LIST[t]
                    SUM_ALL_T += a * ht
                  
            # H of T PRIME        

            htp = return_label(TREES[TREES_INDEX], Input_Vector[m])
            y = Input_Vector[m][TEST_LABEL]
            if htp == y:
                topsum += math.exp(-1 * y * SUM_ALL_T)
            else: #htp != y:
                bottomsum += math.exp(-1 * y * SUM_ALL_T)
        # NOW UPDATE ALPHA
        ALPHA_LIST[TREES_INDEX] = .5 * math.log1p(float(topsum)/float(bottomsum))
        
        return ALPHA_LIST
        
def accuracy(TREE, Input_Vector):
    
    total_true = 0
    total = 0
    for m in range(len(Input_Vector)):
        ht = return_label(TREE, Input_Vector[m])
        if ht == Input_Vector[m][TEST_LABEL]:
            total_true += 1
        total += 1
    return float(total_true) / float(total)
    
def make_trees():
    TREES = []
    
    for LEFT in range(2):
        for RIGHT in range(2):
            for ATTRIBUTE in range(23):
                if ATTRIBUTE !=  TEST_LABEL:
                    TREES.append([LEFT, RIGHT, ATTRIBUTE])
    return TREES

def make_alphas(size):
    alphas = [1.0/size] * size
    return alphas

def return_label(tree, row):
    if row[tree[ATTR]] == 0:
        return tree[0]
    else:
        return tree[1]
    
def calculate_loss(TREES, ALPHA_LIST, Input_Vector):
    LOSS = 0
    for m in range(len(Input_Vector)):
        
        SUM_ALL_T = 0
        for t in range(len(TREES)):
            ht = return_label(TREES[t], Input_Vector[m])
            a = ALPHA_LIST[t]
            SUM_ALL_T += a * ht
        y = Input_Vector[m][TEST_LABEL]
        
        LOSS = math.exp(-1 * y * SUM_ALL_T)
    return LOSS
        
test()