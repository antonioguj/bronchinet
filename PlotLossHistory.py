#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import sys


if( len(sys.argv)<2 ):
    print("ERROR. Please input the loss history file(s) name(s)... EXIT")
    sys.exit(0)

nb_plot_files = len(sys.argv) - 1

if nb_plot_files == 1:

    lossHistoryFile = str(sys.argv[1])

    data = np.loadtxt(lossHistoryFile, skiprows=1)

    plt.plot(data[:,0], data[:,1], color='b', label='Train')
    plt.plot(data[:,0], data[:,2], color='r', label='Valid')

else:
    cmap = plt.get_cmap('rainbow')
    colors = [ cmap(float(i)/(nb_plot_files-1)) for i in range(nb_plot_files) ]

    for i in range(nb_plot_files):

	lossHistoryFile = str(sys.argv[i+1])

	data = np.loadtxt(lossHistoryFile, skiprows=1)

    	plt.plot(data[:,0], data[:,1], color=colors[i], label='Train_%i'%(i))
    	plt.plot(data[:,0], data[:,2], color=colors[i], linestyle='--', label='Valid_%i'%(i))	
    #endfor

plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()
