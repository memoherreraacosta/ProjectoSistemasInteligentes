#------------------------------------------------------------------------------------------------------------------
#   Sample program for data acquisition and recording.
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
import numpy as np

from glob import glob
from sklearn import svm
from pprint import pprint
from scipy.linalg import svd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


# Import Datos misteriosos data set
data_set = glob("data/*")
datos = np.loadtxt(data_set[0], dtype="float")
x = datos[:,1:]
y = datos[:,0]
y = y-1
n_features = len(x[0])


# 5-fold cross-validation
# Evaluate model SVM linear
kf = KFold(n_splits=5, shuffle = True)
clf = svm.SVC(kernel = 'linear')
acc = 0
recall = np.array([0., 0.])
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
    acc_i = (cm[0,0]+cm[1,1])/len(y_test)
    acc += acc_i

    recall[0] += cm[0,0]/ (cm[0,0]+cm[0,1])
    recall[1] += cm[1,1]/ (cm[1,0]+cm[1,1])

print("\nSVM Linear model")
print('ACC SVM Linear = ', acc/5)
print('RECALL SVM Linear = ', recall/5)


# 5-fold cross-validation
# Evaluate model SVM linear
kf = KFold(n_splits=5, shuffle = True)
clf = svm.SVC(kernel = 'rbf')
acc = 0
recall = np.array([0., 0.])
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
    acc_i = (cm[0,0]+cm[1,1])/len(y_test)
    acc += acc_i

    recall[0] += cm[0,0]/ (cm[0,0]+cm[0,1])
    recall[1] += cm[1,1]/ (cm[1,0]+cm[1,1])

print("\nSVM Radial model")
print('ACC SVM Radial= ', acc/5)
print('RECALL SVM Radial= ', recall/5)


# Evaluate model Neuronal Network
kf = KFold(n_splits=5, shuffle = True)
acc = 0
recall = np.array([0., 0.])
for train_index, test_index in kf.split(x):
    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]
    # y_train = np_utils.to_categorical(y_train)

    clf = Sequential()
    clf.add(Dense(8, input_dim=n_features, activation='relu'))
    clf.add(Dense(8, activation='relu'))
    clf.add(Dense(1, activation='sigmoid'))
    clf.compile(loss='binary_crossentropy', optimizer='adam')
    clf.fit(x_train, y_train, epochs=150, batch_size=8, verbose=0)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]
    y_pred = (clf.predict(x_test) > 0.5).astype("int32")

    cm = confusion_matrix(y_test, y_pred)
    acc += (cm[0,0]+cm[1,1])/len(y_test)
    recall[0] += cm[0,0]/(cm[0,0] + cm[0,1])
    recall[1] += cm[1,1]/(cm[1,0] + cm[1,1])

print("\nNeuronal Network")
acc = acc/5
print('ACC Neuronal network= ', acc)
recall = recall/5
print('RECALL Neuronal network= ', recall)


results = dict()
# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle = True)
neighbours_distance = 5
clf = KNeighborsClassifier(n_neighbors=neighbours_distance)
acc = 0
recall = np.array([0., 0.])
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
    acc_i = (cm[0,0]+cm[1,1])/len(y_test)
    acc += acc_i
    recall[0] += cm[0,0]/ (cm[0,0]+cm[0,1])
    recall[1] += cm[1,1]/ (cm[1,0]+cm[1,1])

print("\nKNN model")
print("'K' Neighbors")
pprint({"ACC": acc/5, "RECALL": recall/5})