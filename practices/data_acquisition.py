#------------------------------------------------------------------------------------------------------------------
#   Sample program for data acquisition and recording.
#------------------------------------------------------------------------------------------------------------------
import time
import socket
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd

# Data configuration
n_channels = 5
samp_rate = 256
emg_data = [[] for i in range(n_channels)]
samp_count = 0
win_size = 256

# Socket configuration
UDP_IP = '127.0.0.1'
UDP_PORT = 8000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.01)

# Data acquisition
start_time = time.time()

#Plot
fig, axs = plt.subplots(2,2)
fig.suptitle('Pairs of variables')
plt.show()

ps = 0
while True:
    try:
        data, addr = sock.recvfrom(1024*1024)

        values = np.frombuffer(data)
        ns = int(len(values)/n_channels)
        samp_count+=ns
        ps+=ns

        for i in range(ns):
            for j in range(n_channels):
                emg_data[j].append(values[n_channels*i + j])

        elapsed_time = time.time() - start_time
        if elapsed_time >= 0.1 and samp_count>=win_size:

            window_data = np.array([x[-win_size:] for x in emg_data])

            # Power Spectral Analisis
            power1, freq1 = psd(window_data[0], NFFT = win_size, Fs = samp_rate)
            power2, freq2 = psd(window_data[2], NFFT = win_size, Fs = samp_rate)
            axs[0,0].cla()
            axs[0,1].cla()
            axs[1,0].cla()
            axs[1,1].cla()
            start_time = time.time()
            axs[0,0].plot(window_data[4], window_data[0], color = 'blue', label = 'Canal 1')
            axs[0,1].plot(window_data[4], window_data[2], color = 'green', label = 'Canal 2')
            axs[0,0].set(xlabel="Time(ms)", ylabel='micro V')
            axs[0,0].set_ylim([-20, 20])
            axs[0,1].set(xlabel="Time(ms)", ylabel='micro V')
            axs[0,1].set_ylim([-20, 20])

            start_index = np.where(freq1 >= 4.0)[0][0]
            end_index = np.where(freq1 >= 60.0)[0][0]

            axs[1,0].plot(freq1[start_index:end_index], power1[start_index:end_index], color = "blue")
            axs[1,0].set(xlabel='Hz', ylabel='Power')

            axs[1,1].plot(freq2[start_index:end_index], power2[start_index:end_index], color = 'green')
            axs[1,1].set(xlabel='Hz', ylabel='Power')

            plt.pause(.001)

            print ("Muestras: ", ps)
            print ("Cuenta: ", samp_count)
            print("")
            ps = 0
    except socket.timeout:
        pass

    

