# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 18:49:05 2018

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
#Scaling data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
#NOw this is a dataset provided by sklearn library
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
sc = StandardScaler() #for scaling the data
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

knn = KNeighborsClassifier(n_neighbors = 7, p = 2, metric='minkowski')
knn1 = KNeighborsClassifier(n_neighbors = 5, p = 3, metric='minkowski')
knn.fit(X_train_std, y_train)
knn1.fit(X_train_std, y_train)
print('The accuracy of the Knn classifier on training data is {:.2f}'.format(knn.score(X_train_std, y_train)))
print('The accuracy of the Knn classifier on test data is {:.2f}'.format(knn.score(X_test_std, y_test)))
print('The accuracy of the Knn2 classifier on training data is {:.2f}'.format(knn1.score(X_train_std, y_train)))
print('The accuracy of the Knn2 classifier on test data is {:.2f}'.format(knn1.score(X_test_std, y_test)))