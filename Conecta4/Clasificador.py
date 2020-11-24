import time
import socket
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

class Clasificador:

    def __init__(self):
        self.samp_rate = 256
        self.win_size = 256
        ytest, xtest, self.n_channels, = self.dataAnalisys("data3.txt")
        y, x, self.n_channels, = self.dataAnalisys("data3.txt")
        print("==============================LINEAL")

        kf = KFold(n_splits=5, shuffle=True)
        self.clf = svm.SVC(kernel='linear')

        accp = 0

        self.clf.fit(x, y)
        y_pred = self.clf.predict(xtest)
        cm = confusion_matrix(ytest, y_pred)
        print("cm test", cm)
        accp = self.getAccuracy(cm, len(ytest))
        print("Accuracy is ", accp)
        self.getInput()


    def getAccuracy(self, cm, len_y):
        return sum(
            cm[i, i]
            for i in range(cm.shape[0])
        )/len_y


    def getInput(self):
        # Data configuration
        n_channels = self.n_channels-2
        #print("chanels", n_channels)
        emg_data = [[] for i in range(n_channels)]
        samp_count = 0
        contador = 0
        time_samp = []

        # Socket configuration
        UDP_IP = '127.0.0.1'
        UDP_PORT = 8000
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        sock.settimeout(0.01)

        start_time = time.time()
        while True:
            try:
                data, addr = sock.recvfrom(1024*1024)

                values = np.frombuffer(data)
                ns = int(len(values)/n_channels)
                samp_count += ns
                contador += ns
                time_samp.append(samp_count/self.samp_rate)
                for i in range(ns):
                    for j in range(n_channels):
                        emg_data[j].append(values[n_channels*i + j])

                elapsed_time = time.time() - start_time
                if (elapsed_time > 0.1):
                    start_time = time.time()

                    ventana_actual = emg_data
                    if (contador > self.win_size):

                        chann1 = emg_data[0]
                        chann2 = emg_data[2]

                        x = chann1[-self.win_size:]
                        y = chann2[-self.win_size:]

                        contador = 0

                        contador = 0

                        power, freq = plt.psd(x, NFFT=self.win_size, Fs=self.samp_rate)
                        power2, freq2 = plt.psd(y, NFFT=self.win_size, Fs=self.samp_rate)
                        plt.clf()

                        start_freq = next(
                            x
                            for x, val in enumerate(freq)
                            if val >= 4.0
                        )
                        end_freq = next(
                            x
                            for x, val in enumerate(freq)
                            if val >= 60.0
                        )

                        start_index = np.where(freq >= 4.0)[0][0]
                        end_index = np.where(freq >= 60.0)[0][0]

                        temp1 = []
                        temp2 = []
                        for hz in range(start_index, end_index+1):
                            temp1.append(power[hz])
                            temp2.append(power2[hz])
                        print("prediction", self.clf.predict([temp1+temp2]))
                        return self.clf.predict([temp1+temp2])
                        #time.sleep(0.2)

            except socket.timeout:
                pass

    # Data test

    def dataAnalisys(self, fileName):
        data = np.loadtxt(fileName)
        samps = data.shape[0]
        self.n_channels = data.shape[1]
        
        channPow1 = {}
        channPow2 = {}
        channPowAverage1 = {}
        channPowAverage2 = {}

        # print('Número de muestras: ', data.shape[0])
        # print('Número de canales: ', data.shape[1])
        # print('Duración del registro: ', samps / samp_rate, 'segundos')
        # print(data)

        # Time channel
        timestamp = data[:, 0]

        # Data channels
        chann1 = data[:, 1]
        chann2 = data[:, 3]

        # Mark data
        mark = data[:, 6]

        frequenciesLabels = []

        training_samples = {}
        for i in range(samps):
            if mark[i] > 0:
                print("Marca", mark[i], 'Muestra', i, 'Tiempo', timestamp[i])

                if (mark[i] > 100) and (mark[i] < 200):
                    iniSamp = i
                    condition_id = mark[i]
                elif mark[i] == 200:
                    if not condition_id in training_samples.keys():
                        training_samples[condition_id] = []
                        channPow1[condition_id] = []
                        channPow2[condition_id] = []
                        channPowAverage1[condition_id] = []
                        channPowAverage2[condition_id] = []
                    training_samples[int(condition_id)].append([iniSamp, i])

        # print('Rango de muestras con datos de entrenamiento:', training_samples)

        # Plot data
        for currentMark in training_samples:
            print("You are seeing the mark ", currentMark)
            ini_samp = training_samples[currentMark][2][0]
            end_samp = training_samples[currentMark][2][1]

            x = chann1[ini_samp: end_samp]
            t = timestamp[ini_samp: end_samp]
            y = chann2[ini_samp: end_samp]

            fig, axs = plt.subplots(2)
            axs[0].plot(t, x, label='Canal 1')
            axs[1].plot(t, y, color='red', label='Canal 2')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('micro V')
            plt.legend()
            plt.clf()

            power, freq = plt.psd(x, NFFT=self.win_size, Fs=self.samp_rate)
            power2, freq2 = plt.psd(y, NFFT=self.win_size, Fs=self.samp_rate)
            plt.clf()

            start_freq = next(x for x, val in enumerate(freq) if val >= 4.0)
            end_freq = next(x for x, val in enumerate(freq) if val >= 60.0)
            # print(start_freq, end_freq)
            start_freq2 = next(y for y, val in enumerate(freq) if val >= 4.0)
            end_freq2 = next(y for y, val in enumerate(freq) if val >= 60.0)
            print(start_freq2, end_freq2)

            # print("La frecuencia es", freq)
            # print("El poder es", power)
            start_index = np.where(freq >= 4.0)[0][0]
            end_index = np.where(freq >= 60.0)[0][0]

            plt.plot(
                freq[start_index:end_index],
                power[start_index:end_index],
                label='Canal 1'
            )
            plt.plot(
                freq2[start_index:end_index],
                power2[start_index:end_index],
                color='red',
                label='Canal 2'
            )
            plt.xlabel('Hz')
            plt.ylabel('Power')
            plt.legend()
            plt.clf()

        frequenciesLabels = freq[start_index:end_index]

        for currentMark in training_samples:

            for i in range(len(training_samples[currentMark])):
                ini_samp = training_samples[currentMark][i][0]
                end_samp = ini_samp + self.win_size
                while(end_samp < training_samples[currentMark][i][1]):

                    # Power Spectral Density (PSD) (1 second of training data)

                    x = chann1[ini_samp: end_samp]
                    t = timestamp[ini_samp: end_samp]
                    y = chann2[ini_samp: end_samp]

                    print("You are currently at ", currentMark,
                          "initial samp ", ini_samp, "end_samp ", end_samp)
                    plt.plot(t, x, label='Canal 1')
                    plt.plot(t, y, color='red', label='Canal 2')
                    plt.xlabel('Tiempo (s)')
                    plt.ylabel('micro V')
                    plt.legend()
                    plt.clf()

                    power, freq = plt.psd(x, NFFT=self.win_size, Fs=self.samp_rate)
                    power2, freq2 = plt.psd(y, NFFT=self.win_size, Fs=self.samp_rate)

                    plt.clf()

                    # print("Power ",power, " freq ",freq)

                    # start_freq = next(x for x, val in enumerate(freq) if val >= 4.0)
                    # end_freq = next(x for x, val in enumerate(freq) if val >= 60.0)
                    # print(start_freq, end_freq)

                    start_index = np.where(freq >= 4.0)[0][0]
                    end_index = np.where(freq >= 60.0)[0][0]

                    # 128: 2 - 30
                    # 256: 4 - 60
                    # 512: 8 - 120
                    temp1 = []
                    temp2 = []
                    for hz in range(start_index, end_index+1):
                        temp1.append(power[hz])
                        temp2.append(power2[hz])

                    channPow1[currentMark].append(temp1)
                    channPow2[currentMark].append(temp2)

                    ini_samp = end_samp
                    end_samp = ini_samp + self.win_size

        ytest = []
        xtest = []
        for mark in training_samples:
            for i in range(len(channPow1[mark])):
                xtest.append(channPow1[mark][i]+channPow2[mark][i])
                ytest.append(mark)

        ytest = np.array(ytest)
        xtest = np.array(xtest)

        return ytest, xtest, self.n_channels
