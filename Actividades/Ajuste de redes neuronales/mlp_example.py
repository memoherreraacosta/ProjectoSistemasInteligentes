#------------------------------------------------------------------------------------------------------------------
import numpy as np

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense

# Import IRIS data set
datos = np.loadtxt("Datos misteriosos.txt", dtype="float")
x = datos[:,1:]
y = datos[:,0]
y = y-1

n_features = len(x[0])

# Evaluate model
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


acc = acc/5
print('ACC Neural network= ', acc)

recall = recall/5
print('RECALL Neural network= ', recall)

# 5-fold cross-validation
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


print('ACC SVM= ', acc/5)

print('RECALL SVM= ', recall/5)

