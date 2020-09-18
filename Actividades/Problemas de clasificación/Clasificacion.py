#------------------------------------------------------------------------------------------------------------------
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import time

# Import IRIS data set
iris = datasets.load_iris()
x = iris.data
y = iris.target
features = iris.feature_names
n_features = len(features)

# Plot pairs of variables
fig, axs = plt.subplots(2,3)
fig.suptitle('Pairs of variables')

axs[0,0].scatter(x[:,0], x[:,1], c = y, cmap=plt.cm.Set1, edgecolors='k')
axs[0,0].set(xlabel=features[0], ylabel=features[1])

axs[0,1].scatter(x[:,0], x[:,2], c = y, cmap=plt.cm.Set1, edgecolors='k')
axs[0,1].set(xlabel=features[0], ylabel=features[2])

axs[0,2].scatter(x[:,0], x[:,3], c = y, cmap=plt.cm.Set1, edgecolors='k')
axs[0,2].set(xlabel=features[0], ylabel=features[3])

axs[1,0].scatter(x[:,1], x[:,2], c = y, cmap=plt.cm.Set1, edgecolors='k')
axs[1,0].set(xlabel=features[1], ylabel=features[2])

axs[1,1].scatter(x[:,1], x[:,3], c = y, cmap=plt.cm.Set1, edgecolors='k')
axs[1,1].set(xlabel=features[1], ylabel=features[3])

axs[1,2].scatter(x[:,2], x[:,3], c = y, cmap=plt.cm.Set1, edgecolors='k')
axs[1,2].set(xlabel=features[2], ylabel=features[3])

# ¿Qué representan las variables incluidas en la base de datos?
# longitud y anchura del tallo y longitud y anchura del petalo
# ¿Consideras que las variables predictoras tienen información suficiente para determinar la clase de cada uno de los tipos de datos?
# En la mayoría de los casos si pero hay casos en los que las variables presentes no son suficientes para encontrar una diferencia.

plt.show()

# Train SVM classifier with all the available observations
clf = svm.SVC(kernel = 'linear')
clf.fit(x, y)

# Predict one new sample
new_ob = []
for i in range(7):
    new_ob.append([random.randint(3,8)+random.random(),random.randint(2,4)+random.random(),random.randint(1,7)+random.random(),random.randint(0,2)+random.random()])
print("New Observation: " + str(new_ob))
print("Prediction for a new observation", clf.predict(new_ob))

# 5-fold cross-validation
kf = KFold(n_splits=10, shuffle = True)
clf = svm.SVC(kernel = 'linear')

acc = 0
precision1 = 0
precision2 = 0
precision3 = 0
recall1 = 0
recall2 = 0
recall3 = 0
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
    # print('acc = ', acc_i)
    acc += acc_i

    precision1_i = cm[0,0]/ (cm[0,0]+cm[1,0] + cm[2, 0])
    precision2_i = cm[1,1]/ (cm[1,1]+cm[0,1] + cm[2, 1])
    precision3_i = cm[2,2]/ (cm[2,2]+cm[1,2] + cm[0, 2])
    precision1 += precision1_i
    precision2 += precision2_i
    precision3 += precision3_i

    recall1_i = cm[0,0]/ (cm[0,0]+cm[0,1] + cm[0, 2])
    recall2_i = cm[1,1]/ (cm[1,0]+cm[1,1] + cm[1, 2])
    recall3_i = cm[2,2]/ (cm[2,0]+cm[2,1] + cm[2, 2])
    recall1 += recall1_i
    recall2 += recall2_i
    recall3 += recall3_i

print('ACC = ', acc/10)
print('PRECISION1 = ', precision1/10)
print('PRECISION2 = ', precision2/10)
print('PRECISION3 = ', precision3/10)

print('RECALL1 = ', recall1/10)
print('RECALL2 = ', recall2/10)
print('RECALL3 = ', recall3/10)

