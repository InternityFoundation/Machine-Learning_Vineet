# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 18:38:43 2018

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #required for plotting the labels
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
x = pd.DataFrame(iris.data) #converting data into pandas dataframe
x.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] #renaming the columns
y = iris.target

a = KMeans(n_clusters=3, init='random')
a.fit(x)
#print(a.labels_) #it will give us the labels
#colormap = np.array(['Red', 'Blue', 'Green'])
#z = plt.scatter(x.sepal_length, x.sepal_width, x.petal_length, c=colormap[a.labels_])
print(accuracy_score(y,a.labels_))
