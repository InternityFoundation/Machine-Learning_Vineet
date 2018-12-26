# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 19:09:17 2018

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
#Scaling data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
#NOw this is a dataset provided by sklearn library
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data
y = iris.target
#Here if we tak different value of random state, then also accuracy shows a change.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
sc = StandardScaler() #for scaling the data
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Create tree object
decision_tree = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=2, min_samples_leaf=2)
decision_tree1 = tree.DecisionTreeClassifier(criterion='entropy')

#Train DT based on scaled training set
decision_tree.fit(X_train_std, y_train)
decision_tree1.fit(X_train_std, y_train)

#Print performance
print('The accuracy of the Decision Tree classifier on training data is {:.2f}'.format(decision_tree.score(X_train_std, y_train)))
print('The accuracy of the Decision Tree classifier on test data is {:.2f}'.format(decision_tree.score(X_test_std, y_test)))

print('The accuracy of the Decision Tree1 classifier on training data is {:.2f}'.format(decision_tree1.score(X_train_std, y_train)))
print('The accuracy of the Decision Tree1 classifier on test data is {:.2f}'.format(decision_tree1.score(X_test_std, y_test)))

