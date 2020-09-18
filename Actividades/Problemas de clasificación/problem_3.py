#------------------------------------------------------------------------------------------------------------------
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob

# Import IRIS data set
# iris = datasets.load_iris()
# x = iris.data
# y = iris.target
# features = iris.feature_names
# n_features = len(features)

# Load Mysterious DataBase
# myst = np.genfromtxt('mysterious.txt',delimiter=" ")
# myst = datasets.load_files('training/')

# Pandas try
# myst = pd.read_csv('mysterious.txt')

# i= 0
# for line in myst:
#     i = i + 1
#     print(i)

files = glob("*.txt")
with open(files[0], "r") as f:
    data = [
        register.split()[1:]
        for register
        in f.readlines()
    ]

print(data)

# data = myst.ix[:,:-1]
# print("Data",data)
x = myst.data
y = myst.data
# features = myst.feature_names
# n_features = len(features)

# Plot pairs of variables
plt.scatter(x[:,1], x[:,2], c = y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel(features[1])
plt.ylabel(features[2])
plt.show()

# Train SVM classifier with all the available observations
clf = svm.SVC(kernel = 'linear')
clf.fit(x, y)

# Predict one new sample
print("Prediction for a new observation", clf.predict( [[1.,2.,3.,4.]] ))

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle = True)
clf = svm.SVC(kernel = 'linear')

acc = 0
for train_index, test_index in kf.split(x):

    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]
    clf.fit(x_train, y_train)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]    
    y_pred = clf.predict(x_test)

    # Calculate confusion matrix and model performance
    cm = confusion_matrix(y_test, y_pred)
    acc_i = (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)    
    print('acc = ', acc_i)

    acc += acc_i 

acc = acc/5
print('ACC = ', acc)