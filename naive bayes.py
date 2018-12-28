import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #required for plotting the labels
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

iris = datasets.load_iris()
x = iris.data
y = iris.target
model = GaussianNB()
model.fit(x,y)
predicted = model.predict(x)

print(metrics.classification_report(y, predicted))
print(metrics.confusion_matrix(y, predicted))