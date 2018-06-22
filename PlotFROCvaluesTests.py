#!/usr/bin/python

from CommonUtil.FunctionsUtil import *
import matplotlib.pyplot as plt
from collections import *
import numpy as np
import sys


if( len(sys.argv)<2 ):
    print("ERROR. Please input the FROC values tests file(s) name(s)... EXIT")
    sys.exit(0)

num_data_files = len(sys.argv)-1

print("Plot FROC values from %s test files:..." %(num_data_files))
print(', '.join(map(lambda item: '\''+item+'\'', sys.argv[1:])))


threshold_list   = []
sensitivity_list = []
FPaverage_list   = []

for i in range(num_data_files):

    data_file = str(sys.argv[i+1])

    data_this = np.loadtxt(data_file, skiprows=2)

    threshold_list  .append(data_this[:, 0])
    sensitivity_list.append(data_this[:, 1])
    FPaverage_list  .append(data_this[:, 2])
#endfor


mark_value = 0.5

labels = ['wBEC_rand',
          'dice_rand',
          'dice_elas']

if num_data_files == 1:
    plt.plot(FPaverage_list[0], sensitivity_list[0], 'o-', color='b')

    # annotate thresholds
    if threshold_list[0] is not None:
        # round thresholds
        threshold_list = ['%.2f' % elem for elem in threshold_list[0]]
        xy_buffer = None
        for i, xy in enumerate(zip(FPaverage_list[0], sensitivity_list[0])):
            if xy != xy_buffer:
                plt.annotate(str(threshold_list[i]), xy=xy, textcoords='data')
                xy_buffer = xy

    plt.xlabel('FalsePositives Average')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.show()

else:
    cmap = plt.get_cmap('rainbow')
    colors = [ cmap(float(i)/(num_data_files-1)) for i in range(num_data_files) ]

    for i in range(num_data_files):
        plt.plot(FPaverage_list[i], sensitivity_list[i], color=colors[i], label=labels[i])

        # find the value corresponding to threshold 0.5 and mark it in plot
        index_mark_value = np.where(threshold_list[i] == mark_value)[0]

        if index_mark_value:
            index_mark_value = index_mark_value[0]
        else:
            #find closest value to threshold 0.5 and mark it
            index_mark_value = np.argmin(np.abs(threshold_list[i] - mark_value))

        plt.scatter(FPaverage_list[i][index_mark_value], sensitivity_list[i][index_mark_value], marker='o', color=colors[i])
    # endfor

    plt.xlabel('False Positive Average')
    plt.ylabel('True Positive Rate')
    plt.xlim([0, 400000])
    plt.title('ROC curve')
    plt.legend(loc='right')
    plt.show()