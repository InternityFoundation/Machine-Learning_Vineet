# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 22:31:04 2018

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
linreg = LinearRegression()
logreg = LogisticRegression()
linreg.fit(X_train, y_train)
logreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
y_pred1 = logreg.predict(X_test)
#print(y_pred)
print(y_pred1)
# compute the RMSE of our predictions
#print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
