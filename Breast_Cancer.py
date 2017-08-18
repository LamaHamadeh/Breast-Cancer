#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:20:40 2017

@author: lamahamadeh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style #style
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#load/clean/organise the data

df = pd.read_csv('/Users/lamahamadeh/Desktop/breast-cancer-data.txt')
df.columns = ['id', 'Clump_Thick', 'Uniform_Cell_Size', 'Uniform_Cell_Shape', 
              'Mar_Adhesion', 'Single_Epi_Cell_Size', 'Bare_Nuclei', 
              'Bland_Chromatin', 'Norm_Nucleoli', 'Mitoses', 'Class']
              
df.replace('?', -9999, inplace = True) #here we put -9999 to let the algorithm 
#treat the missing values as outliers. Other way is to use df.dropna to get 
#rid of them. This depends on your preference and on your data.  

print (df.dtypes)
#it can be seen that the 'nuclei' column has an 'object' type where it has to be 'int'. Therefore, we need to change it to numeric value
df.Bare_Nuclei = pd.to_numeric(df.Bare_Nuclei, errors = 'coerce')

df.drop(['id'], axis = 1, inplace = True) #this is an unwanted data and remove it
#will increase the effiency and the accuracy of the classifier       

print(df.head())
#-------------------------------

#apply the K nearest neighbor classifier
#seperate the data to train and test parts uisng cross_validation
X = np.array(df.drop(['Class'], axis = 1)) #the rest of the dataset apart from 
#the label column
Y = np.array(df['Class']) #the label of the dataset



X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size=0.5, random_state = 7)

#apply the knn method
Knn = KNeighborsClassifier(n_neighbors = 2)
#train the data
Knn.fit(X_train,Y_train)
#test the data
accuracy = Knn.score(X_test, Y_test)#this to see how accurate the algorithm is in terms 
#of defining the tumor to be either melignant or bengin

print('accuracy of the model is: ', accuracy)

#-------------------------------


#prediction
#if I want to predict the class of only one measurement
example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
example_measures = example_measures.reshape(1,-1)
prediction = Knn.predict(example_measures)
print('Prediction of the measures is: ', prediction)

#if I have multiple measurements I need to predict their class
example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1],[4, 2, 1, 1, 1, 2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures),-1)
prediction = Knn.predict(example_measures)
print('Prediction of the measures is: ', prediction)


#-------------------------------
# Plotting and visualisation

print ("Plotting...")