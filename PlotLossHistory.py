#!/usr/bin/python

from CommonUtil.FunctionsUtil import *
import matplotlib.pyplot as plt
from collections import *
import numpy as np
import sys


if( len(sys.argv)<2 ):
    print("ERROR. Please input the loss history file(s) name(s)... EXIT")
    sys.exit(0)

num_plot_files = len(sys.argv)-1

print("Plot loss history from %s files:..." %(num_plot_files))
print(', '.join(map(lambda str: '\''+str+'\'', sys.argv[1:])))


data_fields_lossHistory_files = OrderedDict()
data_fields_lossHistory_files['epoch'] = []
data_fields_lossHistory_files['loss'] = []

for i in range(num_plot_files):

    lossHistory_file = str(sys.argv[i+1])

    with open(lossHistory_file, 'r') as infile:
        header_line = infile.readline()
        header_lossHistory = map(lambda str: str.replace('/','').replace('\n',''), header_line.split(' '))

    data_lossHistory = np.loadtxt(lossHistory_file, skiprows=1)


    # run checks correct data format
    num_cols_header = len(header_lossHistory)
    num_cols_data   = data_lossHistory.shape[1]

    if (num_cols_header != num_cols_data):
        print("ERROR. Format input file not correct... EXIT")
        sys.exit(0)
    if ('epoch' not in header_lossHistory) or ('loss' not in header_lossHistory):
        print("ERROR. mandatory fields \'epoch\' or \'loss'\ not found in file \'%s\': ... EXIT" %(lossHistory_file))
        sys.exit(0)
    # check that every field has in 'val_%' associated
    headers_stdalone_fields = list(filter(lambda str: (str!='epoch') and (str[0:4]!='val_'), header_lossHistory))
    for name in headers_stdalone_fields:
        val_name = 'val_'+name
        if val_name not in header_lossHistory:
            print("ERROR. not found the validation value \'val_\' assiciated to the field \'%s\': ... EXIT" % (name))
            sys.exit(0)


    data_fields_lossHistory_this = OrderedDict(zip(header_lossHistory, np.transpose(data_lossHistory)))

    # add mandatory fields: 'epoch' and 'loss'
    data_fields_lossHistory_files['epoch'].append(data_fields_lossHistory_this.pop('epoch'))
    data_fields_lossHistory_files['loss' ].append([data_fields_lossHistory_this.pop('loss'),
                                                   data_fields_lossHistory_this.pop('val_loss')])


    # Look for additional fields in loss History file:
    # get keys of additional fields already found in previous files:
    keys_extra_fields_existing = list(filter(lambda str: (str!='epoch') and (str!='loss'), data_fields_lossHistory_files.keys()))

    for (key, val) in data_fields_lossHistory_this.iteritems():
        if key[0:4]=='val_':
            continue

        if key not in keys_extra_fields_existing:
            # new extra field found: allocate in dictionary
            # create empty spaces to account for previous files
            data_fields_lossHistory_files[key] = [None] * i  # x2 (train and val)
        else:
            keys_extra_fields_existing.remove(key)

        val_key = 'val_' + key
        # add new data, corresponding to both train and validation data
        data_fields_lossHistory_files[key].append([val, data_fields_lossHistory_this[val_key]])
    #endfor

    # for existing extra fields that are not in this file, add empty spaces
    for key in keys_extra_fields_existing:
        data_fields_lossHistory_files[key].append(None)
#endfor


print("Found fields to plot loss history of: %s..." %(', '.join(map(lambda str: '/'+str+'/', data_fields_lossHistory_files.keys()))))

epochs = data_fields_lossHistory_files.pop('epoch')

# run checks correct data format
for (key, data) in data_fields_lossHistory_files.iteritems():
    # for each key the data dim must be as much as num plot files,
    # including empty lists for files without the key
    if len(data) != num_plot_files:
        print("ERROR. for key \'%s\' the data dimension is not correct: \'%s\'...EXIT" % (key, len(data)))
        sys.exit(0)
    # for each file, the data must have two lists (for train and validation data)
    for i, data_row in enumerate(data):
        if data_row and len(data_row) != 2:
            print("ERROR. for key \'%s\' the data dimension for file \'%s\' is not correct: \'%s\'...EXIT" % (key, i, len(data_row)))
            sys.exit(0)


for (key, data) in data_fields_lossHistory_files.iteritems():

    num_files_plot_data = len(data)
    if num_files_plot_data == 1:

        plt.plot(epochs[0], data[0][0], color='b', label='train')
        plt.plot(epochs[0], data[0][1], color='r', label='valid')

    else:
        cmap = plt.get_cmap('rainbow')
        colors = [ cmap(float(i)/(num_files_plot_data-1)) for i in range(num_files_plot_data) ]

        for i in range(num_files_plot_data):
            # skip files that do not contain this data
            if data[i]:
                plt.plot(epochs[i], data[i][0], color=colors[i], label='train_%i'%(i))
                plt.plot(epochs[i], data[i][1], color=colors[i], linestyle='--', label='valid_%i'%(i))
        #endfor

    plt.xlabel('epoch')
    plt.ylabel(str(key))
    plt.legend(loc='best')
    plt.show()