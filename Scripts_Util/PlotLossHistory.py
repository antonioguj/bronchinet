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
from collections import OrderedDict
import numpy as np
import argparse



def main(args):

    if args.fromfile:
        if not isExistfile(args.listinputfiles):
            message = "File \'%s\' not found..." %(args.listinputfiles)
            CatchErrorException(message)
        fout = open(args.listinputfiles, 'r')
        list_input_files = [infile.replace('\n','') for infile in fout.readlines()]
        print("\'inputfiles\' = %s" % (list_input_files))
    else:
        list_input_files = [infile.replace('\n','') for infile in args.inputfiles]
    num_input_files = len(list_input_files)

    print("Files to plot the loss history from: \'%s\'..." %(num_input_files))
    for i, ifile in enumerate(list_input_files):
        print("%s: \'%s\'" %(i+1, ifile))
    #endfor


    # ---------- SETTINGS ----------
    labels_train = ['train_%i'%(i+1) for i in range(num_input_files)]
    labels_valid = ['valid_%i'%(i+1) for i in range(num_input_files)]
    #labels_train = ['loss', 'loss_avrg20ep', 'loss_avrg50ep']
    #labels_valid = labels_train

    #cmap = plt.get_cmap('rainbow')
    #colors = [cmap(float(i)/(num_input_files-1)) for i in range(num_input_files)]
    colors = ['blue', 'red', 'green', 'yellow', 'orange']
    # ---------- SETTINGS ----------



    dict_data_losses_fields_files = OrderedDict()
    dict_data_losses_fields_files['epoch'] = []
    dict_data_losses_fields_files['loss']  = []

    for i, in_file in enumerate(list_input_files):

        raw_data_this_string = np.genfromtxt(in_file, dtype=str, delimiter=' ')
        raw_data_this_float  = np.genfromtxt(in_file, dtype=float, delimiter=' ')

        header_this = list(raw_data_this_string[0, :])
        list_fields = [elem.replace('/','') for elem in header_this]
        data_this   = raw_data_this_float[1:, :]

        # check for correct loss history format
        if (list_fields[0] != 'epoch') or ('loss' not in list_fields) or ('val_loss' not in list_fields):
            message = 'mandatory fields \'epoch\', \'loss\', \'val_loss\' not found in file \'%s\'' %(in_file)
            CatchErrorException(message)

        index_field_loss     = list_fields.index('loss')
        index_field_val_loss = list_fields.index('val_loss')

        dict_data_losses_fields_files['epoch'].append(data_this[:,0])
        dict_data_losses_fields_files['loss'] .append([data_this[:,index_field_loss],
                                                       data_this[:,index_field_val_loss]])


        # check whether there are extra fields, and that each has two columns: +1 for validation (with preffix 'val_%')
        list_extra_fields        = [elem for elem in list_fields if elem not in ['epoch', 'loss', 'val_loss']]
        list_extra_fields_novals = [elem for elem in list_extra_fields if (elem[0:4]!='val_')]

        for i, i_field in enumerate(list_extra_fields_novals):
            i_valid_field = 'val_' + i_field
            if i_valid_field not in list_fields:
                message = 'for the field \'%s\', the associated validation field \'%s\' not found' % (i_field, i_valid_field)
                CatchErrorException(message)

            # check whether extra field exists in the dict for output losses. If not, start new elem with empty list
            if i_field not in dict_data_losses_fields_files.keys():
                dict_data_losses_fields_files[i_field] = []

            index_field     = list_fields.index(i_field)
            index_field_val = list_fields.index(i_valid_field)

            dict_data_losses_fields_files[i_field].append([data_this[:, index_field],
                                                           data_this[:, index_field_val]])
        # endfor
    # endfor

    epochs_files = dict_data_losses_fields_files.pop('epoch')
    print("Found fields to plot loss history: %s..." %(dict_data_losses_fields_files.keys()))



    for (ifield, data_files) in dict_data_losses_fields_files.iteritems():
        num_data_files = len(data_files)

        if num_data_files == 1:
            plt.plot(epochs_files[0], data_files[0][0], color='b', label='train')
            plt.plot(epochs_files[0], data_files[0][1], color='r', label='valid')
            plt.xlabel('Epoch')
            plt.ylabel(ifield.title())
            plt.ylim([0.0, 1.0])
            plt.legend(loc='best')
            plt.show()

        else:
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            for i in range(num_data_files):
                axs[0].plot(epochs_files[i], data_files[i][0], color=colors[i], label=labels_train[i])
                axs[1].plot(epochs_files[i], data_files[i][1], color=colors[i], label=labels_valid[i])
            #endfor
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel(ifield.title())
            axs[0].set_title('Training')
            axs[0].legend(loc='best')
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel(ifield.title())
            axs[1].set_title('Validation')
            axs[1].legend(loc='best')
            plt.show()
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfiles', type=str, nargs='*')
    parser.add_argument('--fromfile', type=bool, default=False)
    parser.add_argument('--listinputfiles', type=str, default='listinputfiles.txt')
    args = parser.parse_args()

    if args.fromfile and not args.listinputfiles:
        message = 'need to input \'listinputfiles\' with filenames to plot'
        CatchErrorException(message)

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
