#------------------------------------------------------------------------------------------------------------------
#   Sample program for data acquisition and recording.
#------------------------------------------------------------------------------------------------------------------
import time
import socket
import numpy as np
import matplotlib.pyplot as plt
import math

# Data configuration
n_channels = 5
win_size = 256
samp_rate = 256
emg_data = [[] for i in range(n_channels)]
samp_count = 0

# Socket configuration
UDP_IP = '127.0.0.1'
UDP_PORT = 8000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.01)

# Data acquisition
start_time = time.time()

fig, axs = plt.subplots(2)

while True:
    try:
        data, addr = sock.recvfrom(1024*1024)                        
            
        values = np.frombuffer(data)
        ns = int(len(values)/n_channels)
        samp_count += ns      

        for i in range(ns):
            for j in range(n_channels):
                emg_data[j].append(values[n_channels*i + j])

        elapsed_time = time.time() - start_time
        if (elapsed_time > 0.1):
            start_time = time.time()
            print ("Muestras: ", ns)
            print ("Cuenta: ", samp_count)
            print ("Última lectura: ", [row[samp_count-1] for row in emg_data])
            print("")

            axs[0].plot(elapsed_time, values[1])
            axs[1].plot(elapsed_time, values[3])
            plt.xlabel("Tiempo (s)")
            plt.ylabel("microvolts")
            plt.pause(0.05)
            #x = values[3]
        '''
        f = [2*math.cos((2*math.pi/samp_rate)*(5) * x) + 4*math.sin((2*math.pi/samp_rate)*(40) * x) for x in range(win_size)]
        power, freq = plt.psd(f, NFFT = win_size, Fs = samp_rate)
        plt.clf()
        start_index = np.where(freq >= 4.0)[0][0]
        end_index = np.where(freq >= 60.0)[0][0]

        plt.plot(freq[start_index:end_index], power[start_index:end_index])
        plt.xlabel('Hz')
        plt.ylabel('Power')   
        '''
    except socket.timeout:
        pass  

plt.show()