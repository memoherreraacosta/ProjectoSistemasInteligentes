#------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Import Data misteriosa data set
datos = np.loadtxt("Datos misteriosos.txt", dtype="float")
x = datos[:,1:]
y = datos[:,0]
y = y-1
n_features = len(x[0])

results = dict()
for i in range(1,11):
    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle = True)
    clf = KNeighborsClassifier(n_neighbors=i)
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

    results[i] = {"ACC": acc/5, "RECALL": recall/5}
    plt.plot(results[i]["RECALL"])
    plt.title('Recall')
    
plt.show()
print("'K' Neighbors")
pprint(results)