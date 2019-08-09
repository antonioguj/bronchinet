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

    if args.fromfile:
        if not isExistfile(args.listinputfiles):
            message = "File \'%s\' not found..." %(args.listinputfiles)
            CatchErrorException(message)
        fout = open(args.listinputfiles, 'r')
        list_input_files = [infile.replace('\n','') for infile in fout.readlines()]
        print("\'inputfiles\' = %s" % (list_input_files))
    else:
        list_input_files = [infile.replace('\n','') for infile in args.inputfiles]
    num_data_files = len(list_input_files)

    print("Files to plot (\'%s\')..." %(num_data_files))
    for i, ifile in enumerate(list_input_files):
        print("%s: \'%s\'" %(i+1, ifile))
    #endfor

    labels = ['model_%i'%(i+1) for i in range(num_data_files)]
    #labels = ['Unet-lev3', 'Unet-lev5', 'UnetGNN-RegAdj', 'UnetGNN-DynAdj']


    data_fields_files = OrderedDict()
    data_cases_files = OrderedDict()
    fields_names = []
    cases_names = []

    for i in range(num_data_files):
        data_file_this = list_input_files[i]
        raw_data_this_string = np.genfromtxt(data_file_this, dtype=str, delimiter=',')
        raw_data_this_float = np.genfromtxt(data_file_this, dtype=float, delimiter=',')

        fields_names_this = [item.replace('/','') for item in raw_data_this_string[0,1:]]
        cases_names_this = [item.replace('\'','') for item in raw_data_this_string[1:,0]]
        data_this = raw_data_this_float[1:,1:]
        num_fields_this = len(fields_names_this)
        num_cases_this = len(cases_names_this)

        if i==0:
            fields_names = fields_names_this
            cases_names = cases_names_this
            for (i, key) in enumerate(fields_names):
                data_fields_files[key] = []
            for (i, key) in enumerate(cases_names):
                data_cases_files[key] = []
        else:
            if fields_names_this != fields_names:
                message = 'fields found in file \'%s\' do not match those found previously: \'%s\'' %(data_file, fields_names)
                CatchErrorException(message)
            #if cases_names_this != cases_names:
                #message = 'fields found in file \'%s\' do not match those found previously: \'%s\'' %(data_file, cases_names)
                #CatchErrorException(message)

        # store data for fields in dictionary
        for (i, key) in enumerate(fields_names):
            data_fields_files[key].append(data_this[:,i])
        # store data for cases in dictionary
        for (i, key) in enumerate(cases_names):
            data_cases_files[key].append(data_this[i,:])
    #endfor

    print("Found fields to plot: \'%s\'..." %(', '.join(map(lambda item: '/'+item+'/', fields_names))))
    print("Found cases to plot: \'%s\'..." %(', '.join(map(lambda item: '/'+item+'/', cases_names))))


    for i, (key, data) in enumerate(data_fields_files.iteritems()):
        plt.boxplot(data, labels=labels)
        plt.title(str(key))
        plt.show()
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

    if args.fromfile and not args.listinputfiles:
        print("ERROR. Input input file name with files to plot...")

    main(args)
