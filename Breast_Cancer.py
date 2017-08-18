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
#Plotting and visualisation (focus on only two features from the dataset)

X1 = np.array(df[['Single_Epi_Cell_Size','Mar_Adhesion']]) #choose only two features
Y = np.array(df['Class']) #the label of the dataset

h = .02  # step size in the mesh
 
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])


# apply Neighbours Classifier and fit the data.
X_train, X_test, Y_train, Y_test = train_test_split (X1, Y, test_size=0.5, random_state = 7)
Knn = KNeighborsClassifier(n_neighbors = 15)
Knn.fit(X1, Y)
 
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = Knn.predict(np.c_[xx.ravel(), yy.ravel()])
 
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
 
# Plot also the training points
plt.scatter(X1[:, 0], X1[:, 1], c=Y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Single_Epi_Cell_Size')
plt.ylabel('Mar_Adhesion')
plt.title('K = 15')

plt.show()
#-------------------------------
