#!/usr/bin/python

from CommonUtil.FunctionsUtil import *
import matplotlib.pyplot as plt
from collections import *
import numpy as np
import sys


if( len(sys.argv)<2 ):
    print("ERROR. Please input the predict accuracy tests file(s) name(s)... EXIT")
    sys.exit(0)

num_data_files = len(sys.argv)-1

print("Plot predict accuracy tests from %s files:..." %(num_data_files))
print(', '.join(map(lambda item: '\''+item+'\'', sys.argv[1:])))


data_fields_files  = OrderedDict()
data_samples_files = OrderedDict()
fields_names  = []
samples_names = []

for i in range(num_data_files):

    data_file = str(sys.argv[i+1])

    raw_data_this_string = np.genfromtxt(data_file, dtype=str)
    raw_data_this_float  = np.genfromtxt(data_file, dtype=float)

    fields_names_this  = [item.replace('/','')  for item in raw_data_this_string[0,1:]]
    samples_names_this = [item.replace('\'','') for item in raw_data_this_string[1:,0]]
    data_this          = raw_data_this_float [1:,1:]

    num_fields_this  = len(fields_names_this)
    num_samples_this = len(samples_names_this)


    if i==0:
        fields_names  = fields_names_this
        samples_names = samples_names_this

        for (i, key) in enumerate(fields_names):
            data_fields_files[key] = []
        for (i, key) in enumerate(samples_names):
            data_samples_files[key] = []
    else:
        if fields_names_this != fields_names:
            print("ERROR. fields found for file '%s' do not coincide with those found previously: '%s'... EXIT" %(data_file, fields_names))
            sys.exit(0)
        if samples_names_this != samples_names:
            print("ERROR. tests samples found for file '%s' do not coincide with those found previously: '%s'... EXIT" %(data_file, samples_names))
            sys.exit(0)

    # store data for fields in dictionary
    for (i,key) in enumerate(fields_names):
        data_fields_files[key].append(data_this[:,i])

    # store data for samples in dictionary
    for (i,key) in enumerate(samples_names):
        data_samples_files[key].append(data_this[i,:])
#endfor


print("Found fields to plot: %s..." %(', '.join(map(lambda item: '/'+item+'/', fields_names))))
print("Found samples to plot: %s..." %(', '.join(map(lambda item: '/'+item+'/', samples_names))))


for (key, data) in data_fields_files.iteritems():

    labels = ['model %s'%(i+1) for i in range(len(data))]
    plt.boxplot(data, labels=labels)
    plt.title('Dice Coefficient')
    plt.show()
#endfor
