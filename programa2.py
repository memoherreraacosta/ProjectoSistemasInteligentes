#------------------------------------------------------------------------------------------------------------------
#   Program that simulates the acquisition and transmision of EMG data.
#------------------------------------------------------------------------------------------------------------------
import numpy as np
import socket
import time
import struct
import matplotlib.pyplot as plt


# EMG data
samp_rate = 256

data = np.loadtxt("Izquierda, derecha, cerrado.txt") 
mark = data[:, data.shape[1]-1]
samps = data.shape[0]

data = np.delete(data, data.shape[1]-1, 1)
data = np.delete(data, 0, 1)
n_channels = data.shape[1]

# Socket configuration
UDP_IP = '127.0.0.1'
UDP_PORT = 8000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send data
start_index = 0
end_index = 0
start_time = time.time()


# Tracker
count = 0

# Time Channel
ch_time = data[:, 0]
while True:

    # Calculate elapsed time and current data index
    time.sleep(0.1)
    elapsed_time = time.time() - start_time

    start_index = end_index
    end_index = int(samp_rate*elapsed_time)

    index = start_index % samps;
    ns = end_index - start_index

    # Send data for the calculated range
    if (ns > 0):

        # Build data package
        out_data = []
        for i in range(ns):
            for j in range(n_channels):
                out_data.append(data[index][j])                
                index+=1
                index %= samps

            if (mark[index] != 0):
                print('--------------- Marca:', int(mark[index]), '---------------')

        pack_string = '<' + str(n_channels*ns) + 'd'
        bin_data = struct.pack(pack_string, *out_data)

        # Send data
        sock.sendto(bin_data, (UDP_IP, UDP_PORT))
        plt.plot(elapsed_time,ns)
        plt.xlabel('Time (s)')
        plt.ylabel('Microvolts')
        plt.ylim(-100,100)
        # plt.pause(0.05)

    print("Muestras enviadas:", ns)
    count += ns
    print("Count:",count)
    print("Elapsed Time", elapsed_time)



plt.show()

# fig=plt.figure() 
# plt.axis([0,1000,0,1]) 

# i=0 
# x=list() 
# y=list() 

# while True:
#     temp_y=np.random.random() 
#     x.append(i) 
#     y.append(temp_y) 
#     plt.plot(x,y) 
#     i+=1 
#     plt.show() 