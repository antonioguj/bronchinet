#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.FunctionsUtil import *
import matplotlib.pyplot as plt
from collections import *
import numpy as np
import argparse



def main(args):

    num_plot_files = len(args.infiles)

    print("Plot loss history from \'%s\' files:..." %(num_plot_files))
    print(', '.join(map(lambda item: '\''+item+'\'', sys.argv[1:])))


    data_fields_lossHistory_files = OrderedDict()
    data_fields_lossHistory_files['epoch'] = []
    data_fields_lossHistory_files['loss'] = []

    for i in range(num_plot_files):
        lossHistory_file = str(args.infiles[i])

        with open(lossHistory_file, 'r') as infile:
            header_line = infile.readline()
            header_lossHistory = map(lambda item: item.replace('/','').replace('\n',''), header_line.split(' '))

        data_lossHistory = np.loadtxt(lossHistory_file, skiprows=1)

        # if the file contains only one line, add extra dimension
        if len(data_lossHistory.shape) == 1:
            if len(data_lossHistory.shape) == 1:
                data_lossHistory = np.array([data_lossHistory, data_lossHistory])
                data_lossHistory[1][0] = data_lossHistory[0][0] + 1

        # run checks correct data format
        num_cols_header = len(header_lossHistory)
        num_cols_data = data_lossHistory.shape[1]

        if (num_cols_header != num_cols_data):
            message = 'format input file not correct'
            CatchErrorException(message)
        if ('epoch' not in header_lossHistory) or ('loss' not in header_lossHistory):
            message = 'mandatory fields \'epoch\' or \'loss\' not found in file \'%s\'' %(lossHistory_file)
            CatchErrorException(message)
        # check that every field has in 'val_%' associated
        headers_stdalone_fields = list(filter(lambda item: (item!='epoch') and (item[0:4]!='val_'), header_lossHistory))
        for name in headers_stdalone_fields:
            val_name = 'val_'+name
            if val_name not in header_lossHistory:
                message = 'not found the validation value \'val_\' assiciated to the field \'%s\'' %(name)
                CatchErrorException(message)

        data_fields_lossHistory_this = OrderedDict(zip(header_lossHistory, np.transpose(data_lossHistory)))

        # add mandatory fields: 'epoch' and 'loss'
        data_fields_lossHistory_files['epoch'].append(data_fields_lossHistory_this.pop('epoch'))
        data_fields_lossHistory_files['loss' ].append([data_fields_lossHistory_this.pop('loss'),
                                                       data_fields_lossHistory_this.pop('val_loss')])

        # Look for additional fields in loss History file:
        # get keys of additional fields already found in previous files:
        keys_extra_fields_existing = list(filter(lambda item: (item!='epoch') and (item!='loss'), data_fields_lossHistory_files.keys()))

        for (key, val) in data_fields_lossHistory_this.iteritems():
            if key[0:4]=='val_':
                continue
            if key not in keys_extra_fields_existing:
                # new extra field found: allocate in dictionary
                # create empty spaces to account for previous files
                data_fields_lossHistory_files[key] = [None] * i # x2 (train and val)
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


    print("Found fields to plot loss history of: \'%s\'..." %(', '.join(map(lambda item: '/'+item+'/', data_fields_lossHistory_files.keys()))))
    epochs = data_fields_lossHistory_files.pop('epoch')

    # run checks correct data format
    for (key, data) in data_fields_lossHistory_files.iteritems():
        # for each key the data dim must be as much as num plot files,
        # including empty lists for files without the key
        if len(data) != num_plot_files:
            message = 'for key \'%s\' the data dimension is not correct: \'%s\'' %(key, len(data))
            CatchErrorException(message)
        # for each file, the data must have two lists (for train and validation data)
        for i, data_row in enumerate(data):
            if data_row and len(data_row) != 2:
                message = 'for key \'%s\' the data dimension for file \'%s\' is not correct: \'%s\'' %(key, i, len(data_row))
                CatchErrorException(message)
        #endfor
    #endfor


    for (key, data) in data_fields_lossHistory_files.iteritems():
        num_files_plot_data = len(data)
        if num_files_plot_data == 1:
            plt.plot(epochs[0], data[0][0], color='b', label='train')
            plt.plot(epochs[0], data[0][1], color='r', label='valid')
            plt.xlabel('epoch')
            plt.ylabel(str(key))
            plt.legend(loc='best')
            plt.show()
        else:
            # cmap = plt.get_cmap('rainbow')
            # colors = [cmap(float(i)/(num_files_plot_data-1)) for i in range(num_files_plot_data)]
            colors = ['blue', 'red', 'green', 'yellow', 'orange']

            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            for i in range(num_files_plot_data):
                # skip files that do not contain this data
                if data[i]:
                    axs[0].plot(epochs[i], data[i][0], color=colors[i], label='train_%i'%(i))
                    axs[1].plot(epochs[i], data[i][1], color=colors[i], label='valid_%i'%(i))
            #endfor
            axs[0].set_xlabel('epoch')
            axs[0].set_ylabel(str(key))
            axs[0].legend(loc='best')
            axs[1].set_xlabel('epoch')
            axs[1].set_ylabel(str(key))
            axs[1].legend(loc='best')
            plt.show()
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles', type=str, nargs='+')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
