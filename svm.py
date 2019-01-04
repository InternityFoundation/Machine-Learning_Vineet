from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.datasets import load_svmlight_file as load_svm
from sklearn import datasets
from sklearn.metrics import accuracy_score

df = datasets.load_iris()
x=df.data
y=df.target

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)
clf_imp=svm.SVC(kernel='linear',C=1, degree=3, gamma='auto', coef0=0.0, shrinking=True, 
                probability=False,tol=0.001, 
                cache_size=200, class_weight=None, verbose=False, max_iter=-1)
clf_imp.fit(xtrain,ytrain)
y_pred = clf_imp.predict(xtest)
print(accuracy_score(ytest, y_pred))