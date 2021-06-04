
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import argparse

from common.functionutil import is_exist_file, calc_moving_average
from common.exceptionmanager import catch_error_exception


def main(args):

    if args.fromfile:
        if not is_exist_file(args.list_input_files):
            message = "File \'%s\' not found..." % (args.list_input_files)
            catch_error_exception(message)

        with open(args.list_input_files, 'r') as fout:
            list_input_files = [infile.replace('\n', '') for infile in fout.readlines()]
        print("\'input_files\' = %s" % (list_input_files))
    else:
        list_input_files = [infile.replace('\n', '') for infile in args.input_files]
    num_input_files = len(list_input_files)

    print("Files to plot the loss history from: \'%s\'..." % (num_input_files))
    for i, ifile in enumerate(list_input_files):
        print("%s: \'%s\'" % (i + 1, ifile))
    # endfor

    labels_train = ['train_%i' % (i + 1) for i in range(num_input_files)]
    labels_valid = ['valid_%i' % (i + 1) for i in range(num_input_files)]

    if args.is_move_aver:
        if (len(args.size_aver) == 1) and (num_input_files > 1):
            # only one size of average window provided: apply the same size to all input files
            args.size_aver = args.size_aver * num_input_files

        elif len(args.size_aver) != num_input_files:
            # several sizes of average window provided: check that there are as many as input files
            message = 'number of input sizes of average window (\'%s\') not equal to number of input files (\'%s\')' \
                      % (len(args.size_aver), num_input_files)
            catch_error_exception(message)

    # ******************************

    dict_data_loss_fields_files = OrderedDict()
    dict_data_loss_fields_files['epoch'] = []
    dict_data_loss_fields_files['loss'] = []
    dict_data_loss_fields_files['val_loss'] = []

    for in_file in list_input_files:

        raw_data_this_string = np.genfromtxt(in_file, dtype=str, delimiter=' ')
        raw_data_this_float = np.genfromtxt(in_file, dtype=float, delimiter=' ')

        header_this = list(raw_data_this_string[0, :])
        list_fields = [elem.replace('/', '') for elem in header_this]
        data_this = raw_data_this_float[1:, :]

        # check for correct content of loss history files
        if (list_fields[0:3] != ['epoch', 'loss', 'val_loss']):
            message = 'mandatory fields \'epoch\', \'loss\', \'val_loss\' not found in file \'%s\'' % (in_file)
            catch_error_exception(message)

        index_field_loss = list_fields.index('loss')
        index_field_val_loss = list_fields.index('val_loss')

        dict_data_loss_fields_files['epoch'].append(data_this[:, 0])
        dict_data_loss_fields_files['loss'].append(data_this[:, index_field_loss])
        dict_data_loss_fields_files['val_loss'].append(data_this[:, index_field_val_loss])

        # --------------------

        # check whether there are extra fields
        list_extra_fields = [elem for elem in list_fields if elem not in ['epoch', 'loss', 'val_loss']]
        if len(list_extra_fields) > 0:
            # check that for each extra field, there are two columns (1 more for validation, with prefix 'val_%')
            list_extra_fields_valid = [elem for elem in list_extra_fields if (len(elem) > 3) and (elem[0:4] == 'val_')]
            list_extra_fields_train = [elem for elem in list_extra_fields if elem not in list_extra_fields_valid]

            if len(list_extra_fields_train) != len(list_extra_fields_valid):
                message = 'number of extra fields to plot history from training and validation not equal'
                catch_error_exception(message)

            for i_field_train, i_field_valid in zip(list_extra_fields_train, list_extra_fields_valid):
                if 'val_%s' % (i_field_train) != i_field_valid:
                    message = 'for extra field \'%s\', the validation field \'%s\' found has wrong format. It ' \
                              'should be \'val_%s\' instead' % (i_field_train, i_field_valid, i_field_train)
                    catch_error_exception(message)

                # add extra field to the dict for output losses, if it doesn't exist already
                if i_field_train not in dict_data_loss_fields_files.keys():
                    dict_data_loss_fields_files[i_field_train] = []

                index_field_train = list_fields.index(i_field_train)
                index_field_valid = list_fields.index(i_field_valid)

                dict_data_loss_fields_files[i_field_train].append(data_this[:, index_field_train])
                dict_data_loss_fields_files[i_field_valid].append(data_this[:, index_field_valid])
        # endfor
    # endfor

    # ******************************

    if args.is_move_aver:
        print('Compute the moving average of the losses, with sizes of average window (for each input file): '
              '\'%s\'...' % (args.size_aver))
        for ifield, data_files in dict_data_loss_fields_files.items():
            num_data_files = len(data_files)

            if ifield == 'epoch':
                # for field 'epochs': decrease the num epochs by the size of average window
                for i in range(num_data_files):
                    num_epochs = len(data_files[i])
                    num_aver_epochs = num_epochs - args.size_aver[i] + 1
                    data_files[i] = data_files[i][0:num_aver_epochs]
                # endfor
            else:
                # for other fields: compute the moving average of the list of values
                for i in range(num_data_files):
                    data_files[i] = calc_moving_average(data_files[i], args.size_aver[i])
        # endfor

    # ******************************

    epochs_files = dict_data_loss_fields_files.pop('epoch')

    list_fields_plot_valid = [elem for elem in dict_data_loss_fields_files.keys()
                              if (len(elem) > 3) and (elem[0:4] == 'val_')]
    dict_data_loss_fields_files_valid = OrderedDict()
    for key_field in list_fields_plot_valid:
        dict_data_loss_fields_files_valid[key_field] = dict_data_loss_fields_files.pop(key_field)

    list_fields_plot_train = list(dict_data_loss_fields_files.keys())
    print("Found fields to plot loss history: %s..." % (list_fields_plot_train))

    for (ifield, data_files_train), (_, data_files_valid) in zip(dict_data_loss_fields_files.items(),
                                                                 dict_data_loss_fields_files_valid.items()):
        num_data_files = len(data_files_train)

        if num_data_files == 1:
            plt.plot(epochs_files[0], data_files_train[0], color='b', label='train')
            plt.plot(epochs_files[0], data_files_valid[0], color='r', label='valid')
            plt.xlabel('Epoch')
            plt.ylabel(ifield.title())
            plt.ylim([0.0, 1.0])
            plt.legend(loc='best')
            plt.show()

        else:
            cmap = plt.get_cmap('rainbow')
            colors = [cmap(float(i) / (num_input_files - 1)) for i in range(num_input_files)]

            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            for i in range(num_data_files):
                axs[0].plot(epochs_files[i], data_files_train[i], color=colors[i], label=labels_train[i])
                axs[1].plot(epochs_files[i], data_files_valid[i], color=colors[i], label=labels_valid[i])
            # endfor

            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel(ifield.title())
            axs[0].set_title('Training')
            axs[0].legend(loc='best')
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel(ifield.title())
            axs[1].set_title('Validation')
            axs[1].legend(loc='best')
            plt.show()
    # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', type=str, nargs='*')
    parser.add_argument('--fromfile', type=bool, default=False)
    parser.add_argument('--list_input_files', type=str, default='list_input_files.txt')
    parser.add_argument('--is_move_aver', type=bool, default=False)
    parser.add_argument('--size_aver', type=int, nargs='+', default=None)
    args = parser.parse_args()

    if args.fromfile and not args.list_input_files:
        message = 'need to input \'list_input_files\' with filenames to plot'
        catch_error_exception(message)

    if args.is_move_aver and not args.size_aver:
        message = 'need to input the size of average window \'size_aver\' when computing the moving average'
        catch_error_exception(message)

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
