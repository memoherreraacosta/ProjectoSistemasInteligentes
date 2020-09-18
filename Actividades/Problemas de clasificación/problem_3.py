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

category_1 = []
category_2 = []
# x = [:,1:] datos
# y = [:,0] etiquetas
# x.shape(dimension) algo asi

datos = np.loadtxt("Datos misteriosos.txt",dtype="float")
x = datos[:,1:]
y = datos[:,0]
targets = ["Clase1","Clase2"]
n_clases = len(targets)

features = [str(i) for i in range(len(x[0]))]
n_features = len(x[0])

print("len x",len(x))
print("len y", len(y))
# print(x)

percentage = 0.1
sample_size = round(len(x) * percentage)
sample_indexes = np.random.choice(len(x),sample_size,replace=False)
print("sample size",sample_size,"(",percentage*100,"% of total data )")
print("indexes len",len(sample_indexes))
print("indexes",sample_indexes)

temp_x = []
temp_y = []
for index in sample_indexes:
    temp_x.append(x[index])
    temp_y.append(y[index])

# print("temp x",temp_x[0])
# print("temp y",temp_y)

x = temp_x[:]
y = temp_y[:]

print("y sample",y)
print("x sample",x)

# print(data)
# print(y[0])
# plt.plot(x[0],y)
# plt.show()
# Kersa utils? 
# output_y = np_utils.to_categorical(y)
# print(output_y)
# print(myst[1])
# print(myst)

# files = glob("*.txt")
# with open(files[0], "r") as f:
#     data = [
#         register.split()[1:]
#         for register
#         in f.readlines()
#     ]


# data = myst.ix[:,:-1]
# print("Data",data)
# x = myst.data
# y = myst.data
# features = myst.feature_names
# n_features = len(features)

# Plot pairs of variables
# plt.scatter(x[:,1], x[:,2], c = y, cmap=plt.cm.Set1, edgecolor='k')
# plt.xlabel(features[1])
# plt.ylabel(features[2])
# plt.show()

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

# acc = acc/5
# print('ACC = ', acc)