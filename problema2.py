#------------------------------------------------------------------------------------------------------------------
#   Sample program for data acquisition and recording.
#------------------------------------------------------------------------------------------------------------------

import socket
import random

import numpy as np
import matplotlib.pyplot as plt

from time import time
from sklearn import svm
from sklearn import datasets
from matplotlib.mlab import psd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from pprint import pprint 
# Data configuration
n_channels = 5
samp_rate = 256
emg_data = [[] for _ in range(n_channels)]
samp_count = 0

# Socket configuration
UDP_IP = '127.0.0.1'
UDP_PORT = 8000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.01)

# Data acquisition
start_time = time()

#Plot
fig, axs = plt.subplots(2,2)
fig.suptitle('Pairs of variables')

# Train SVM classifier with all the available observations
clf = svm.SVC(kernel = 'linear')

ps = 0
acc = 0
precision1 = 0
precision2 = 0
precision3 = 0
recall1 = 0
recall2 = 0
recall3 = 0

window_size = 1

while True:
    try:
        data, addr = sock.recvfrom(1024*1024)
        values = np.frombuffer(data)
        ns = len(values) // n_channels
        samp_count += ns
        ps += ns

        for i in range(ns):
            for j in range(n_channels):
                emg_data[j].append(values[n_channels*i + j])

        elapsed_time = time() - start_time
        if (elapsed_time > window_size):
            window_data = np.array([x[samp_count-ps:] for x in emg_data])
            print("SHAPE: {0}\n".format(np.shape(window_data)))
            pprint(window_data)
            # Power Spectral Analisis
            power1, freq1 = psd(window_data[0], NFFT=ps, Fs=samp_rate)
            power2, freq2 = psd(window_data[2], NFFT=ps, Fs=samp_rate)

            axs[0,0].cla()
            axs[0,1].cla()
            axs[1,0].cla()
            axs[1,1].cla()
            start_time = time()
            axs[0,0].plot(window_data[4], window_data[0], color = 'blue', label = 'Canal 1')
            axs[0,1].plot(window_data[4], window_data[2], color = 'green', label = 'Canal 2')
            axs[0,0].set(xlabel="Time(ms)", ylabel='micro V')
            axs[0,0].set_ylim([-200, 200])
            axs[0,1].set(xlabel="Time(ms)", ylabel='micro V')
            axs[0,1].set_ylim([-200, 200])

            start_index = np.where(freq1 >= 4.0)[0][0]
            end_index = np.where(freq1 >= 60.0)[0][0]

            axs[1,0].plot(freq1[start_index:end_index], power1[start_index:end_index], color = "blue")
            axs[1,0].set(xlabel='Hz', ylabel='Power')

            start_index = np.where(freq2 >= 4.0)[0][0]
            end_index = np.where(freq2 >= 60.0)[0][0]

            axs[1,1].plot(freq2[start_index:end_index], power2[start_index:end_index], color = 'green')
            axs[1,1].set(xlabel='Hz', ylabel='Power')

            plt.pause(.01)

            # 5-fold cross-validation
            kf = KFold(n_splits = 5, shuffle = True)
            clf = svm.SVC(kernel = 'linear')

            x = window_data[:,1:]
            y = window_data[:,0]
            y = y-1
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
                # Predict one new sample
                print("Prediction for a new observation", clf.predict(x_test))

            ps = 0
    except socket.timeout:
        pass

plt.show()