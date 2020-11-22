#------------------------------------------------------------
#   Sample program for data acquisition and recording.
#------------------------------------------------------------
import time
import socket
import numpy as np
from matplotlib.mlab import psd
from glob import glob
from sklearn import svm
from sklearn.metrics import confusion_matrix
from pynput.keyboard import Key, Controller


def main():
    #Keyboard
    keyboard = Controller()
    l_key_pressed = False
    n_key_pressed = False

    # Data configuration
    n_channels = 5
    samp_rate = 256
    samp_count = 0
    win_size = 256
    file = 3

    # data training
    clf = training(file, win_size, samp_rate)

    # Socket configuration
    UDP_IP = "127.0.0.1"
    UDP_PORT = 8000
    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(0.01)

    # Data acquisition
    start_time = time.time()

    ps = 0
    while True:
        try:
            data, addr = sock.recvfrom(1024*1024)

            values = np.frombuffer(data)
            ns = int(len(values)/n_channels)
            samp_count+=ns
            ps+=ns

            emg_data = [[] for _ in range(n_channels)]
            for i in range(ns):
                for j in range(n_channels):
                    emg_data[j].append(values[n_channels*i + j])

            elapsed_time = time.time() - start_time
            if elapsed_time >= 0.1 and samp_count>=win_size:

                window_data = np.array([x[-win_size:] for x in emg_data])
                powers = []
                # Power Spectral Analisis
                power1, freq1 = psd(window_data[0], NFFT = win_size, Fs = samp_rate)
                start_index = np.where(freq1 >= 4.0)[0][0]
                end_index = np.where(freq1 >= 60.0)[0][0]
                powers.extend(power1[start_index:end_index])
                power2, freq2 = psd(window_data[2], NFFT = win_size, Fs = samp_rate)
                powers.extend(power2[start_index:end_index])
                pred = int(clf.predict([powers])[0])

                print("Prediccion: ", pred)
                # print ("Muestras: ", ps)
                # print ("Cuenta: ", samp_count)
                print("")
                if(pred == 0 and not l_key_pressed):
                    l_key_pressed = True
                    keyboard.press(Key.right)
                    keyboard.release(Key.right)
                elif(pred == 2):
                    l_key_pressed = False
                    n_key_pressed = False
                elif(pred == 1 and not n_key_pressed):
                    n_key_pressed = True
                    keyboard.press(Key.down)
                    keyboard.release(Key.down)
                ps = 0
        except socket.timeout:
            pass

def preprocessing(file, win_size, samp_rate):
    # Construir X y Y
    files = glob("data/*")
    data = np.loadtxt(files[file])

    samps = data.shape[0]

    # Data channels
    channels = [data[:, i] for i in [1,3]]

    y = data[:, 6]

    training_samples = {}
    for i in range(samps):
        if y[i] > 0:
            # print("Marca", y[i], 'Muestra', i, 'Tiempo', time[i])
            if  (y[i] > 100) and (y[i] < 200):
                iniSamp = i
                condition_id = y[i]-101
            elif y[i] == 200:
                if not condition_id in training_samples.keys():
                    training_samples[condition_id] = []
                training_samples[int(condition_id)].append([iniSamp, i])

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
    return x, y


def training(file, win_size, samp_rate):
    x_train, y_train = preprocessing(file, win_size, samp_rate)
    x_test, y_test = preprocessing(file+1, win_size, samp_rate)

    # 5-fold cross-validation
    # Evaluate model SVM linear
    clf = svm.SVC(kernel = 'linear')

    # Training phase
    clf.fit(x_train, y_train)

    # Test phase
    y_pred = clf.predict(x_test)

    # Calculate confusion matrix and model performance
    recall = []
    cm = confusion_matrix(y_test, y_pred)
    acc = sum([cm[i, i] for i in range(len(cm[0]))])/len(y_test)

    for i in range(len(cm[0])):
        recall.append(cm[i,i]/ sum(cm[i,:]))


    print("\nSVM Linear model")
    print("ACC: ", acc, "Recall: ", recall)
    return clf


if __name__ == '__main__':
    main()