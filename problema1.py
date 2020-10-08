#------------------------------------------------------------------------------------------------------------------
#   Sample program for data acquisition and recording.
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd

from glob import glob
from sklearn import svm
from sklearn import tree
from pprint import pprint
from scipy.linalg import svd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


# Construir X y Y
files = glob("data/*")
data = np.loadtxt(files[0])

win_size = 256 # 1 seg
samp_rate = 256

samps = data.shape[0]

# Time channel
time = data[:, 0]

# Data channels
channels = [data[:, i] for i in [1,3]]
# x, y data
# x = data[:, 1:5]
y = data[:, 6]

training_samples = {}
for i in range(samps):
    if y[i] > 0:
        # print("Marca", y[i], 'Muestra', i, 'Tiempo', time[i])
        if  (y[i] > 100) and (y[i] < 200):
            iniSamp = i
            condition_id = y[i]-100
        elif y[i] == 200:
            if not condition_id in training_samples.keys():
                training_samples[condition_id] = []
            training_samples[int(condition_id)].append([iniSamp, i])

print('Rango de muestras con datos de entrenamiento:', training_samples)

Y = []
X = []
for condition_id, samples in training_samples.items():
    for sample in samples:
        for i in range(sample[0], sample[1] if sample[1]%win_size==0 else sample[1]-win_size, win_size):
            row = []
            for channel in channels:
                ini_samp = i
                end_samp = i + win_size
                x = channel[ini_samp : end_samp]

                power, freq = psd(x, NFFT = win_size, Fs = samp_rate)
                start_index = np.where(freq >= 4.0)[0][0]
                end_index = np.where(freq >= 60.0)[0][0]
                row.extend(power[start_index:end_index])
            X.append(row)
            Y.append(condition_id)

x = np.array(X)
y = np.array(Y)



n_features = x.shape[1]
results =  dict()

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
results["Linear model"] = {"ACC": acc/5, "Recall": recall/5}


# 5-fold cross-validation
# Evaluate model SVM radial
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

results["Radial model"] = {"ACC": acc/5, "Recall": recall/5}


# Decision tree
# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle = True)
clf = tree.DecisionTreeClassifier()
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

results["DTree"] = {"ACC": acc/5, "Recall": recall/5}

# KNN model
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

results["KNN"] = {"ACC": acc/5, "Recall": recall/5}

# Evaluate model Neuronal Network Multi Capa
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

results["red multicapa"] = {"ACC": acc/5, "Recall": recall/5}

pprint(results)

# Evaluate model Neuronal Network Una capa
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
    clf.compile(loss='binary_crossentropy', optimizer='adam')
    clf.fit(x_train, y_train, epochs=150, batch_size=8, verbose=0)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]
    y_pred = (clf.predict(x_test) > 0.5).astype("int32")

    print(y_test)
    print(y_pred)
    cm = confusion_matrix(y_test, y_pred)
    acc += (cm[0,0]+cm[1,1])/len(y_test)
    recall[0] += cm[0,0]/(cm[0,0] + cm[0,1])
    recall[1] += cm[1,1]/(cm[1,0] + cm[1,1])

results["red unacapa"] = {"ACC": acc/5, "Recall": recall/5}


