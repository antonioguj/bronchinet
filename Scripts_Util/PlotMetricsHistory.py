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
    num_plot_files = len(list_input_files)

    print("Files to plot (\'%s\')..." %(num_plot_files))
    for i, ifile in enumerate(list_input_files):
        print("%s: \'%s\'" %(i+1, ifile))
    #endfor

    labels = ['model_%i'%(i+1) for i in range(num_plot_files)]

    cmap = plt.get_cmap('rainbow')
    colors = [cmap(float(i)/num_plot_files) for i in range(num_plot_files)]
    # colors = ['blue', 'red', 'green', 'yellow', 'orange']


    list_epochs = []
    data_fields_files = OrderedDict()
    fields_names = []

    for (i, in_data_file) in enumerate(list_input_files):
        data_this_string = np.genfromtxt(in_data_file, dtype=str, delimiter=',')
        fields_names_this = [item.replace('/','') for item in data_this_string[0,1:]]
        num_fields_this = len(fields_names_this)

        data_this = np.loadtxt(in_data_file, dtype=float, skiprows=1, delimiter=',')
        list_epochs.append(data_this[:, 0])

        if i==0:
            fields_names = fields_names_this
            for (i, key) in enumerate(fields_names):
                data_fields_files[key] = []
        else:
            pass
            # if fields_names_this != fields_names:
            #     message = 'fields found in file \'%s\' do not match those found previously: \'%s\'' %(in_data_file, fields_names)
            #     CatchErrorException(message)

        # store data for fields in dictionary
        for (i, key) in enumerate(fields_names_this):
            data_fields_files[key].append(data_this[:, i+1])
        #endfor
    #endfor

    print("Found fields to plot: \'%s\'..." % (', '.join(map(lambda item: '/' + item + '/', fields_names))))


    for (key, data) in data_fields_files.iteritems():
        num_data_plot = len(data)
        for i in range(num_data_plot):
            plt.plot(list_epochs[i], data[i], color=colors[i], label=labels[i])
            plt.xlabel('epoch')
            plt.ylabel(str(key))
            plt.legend(loc='best')
            plt.show()
        #endfor
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfiles', type=str, nargs='*')
    parser.add_argument('--fromfile', type=bool, default=False)
    parser.add_argument('--listinputfiles', type=str, default='listinputfiles.txt')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)