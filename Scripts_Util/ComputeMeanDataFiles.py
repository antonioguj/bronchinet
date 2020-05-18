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
    num_input_files = len(list_input_files)

    print("Files to compute the mean from: \'%s\'..." %(num_input_files))
    for i, ifile in enumerate(list_input_files):
        print("%s: \'%s\'" %(i+1, ifile))
    #endfor


    if num_input_files == 1:
        print("Compute the mean of fields for cases from a single file...")

        raw_data_string = np.genfromtxt(list_input_files[0], dtype=str, delimiter=', ')
        raw_data_float  = np.genfromtxt(list_input_files[0], dtype=float, delimiter=', ')

        header_file    = list(raw_data_string[0, :])
        rows1elem_file = list(raw_data_string[:, 0])
        data           = raw_data_float[1:, 1:]

        fields_names = [item.replace('/','') for item in header_file[1:]]
        cases_names  = rows1elem_file[1:]

        print("Compute mean data for fields: \'%s\'... and from cases: \'%s\'..." % (fields_names, cases_names))


        # Compute mean of data along the first dimension of array (cases)
        mean_data_cases = np.mean(data, axis=0)

        # Output mean results
        for i, ifield in enumerate(fields_names):
            print("Mean of \'%s\': %0.6f..." %(ifield, mean_data_cases[i]))
        # endfor

    else:
        print("Compute the mean of all data from \'%s\' files: %s..." %(num_input_files, list_input_files))

        for (i, in_file) in enumerate(list_input_files):

            raw_data_this_string = np.genfromtxt(in_file, dtype=str, delimiter=', ')
            raw_data_this_float  = np.genfromtxt(in_file, dtype=float, delimiter=', ')

            header_this   = list(raw_data_this_string[0, :])
            rows1elem_this= list(raw_data_this_string[:, 0])
            data_this     = raw_data_this_float[1:, 1:]

            if i == 0:
                header_file   = header_this
                rows1elem_file= rows1elem_this
                num_rows      = len(rows1elem_file)
                num_cols      = len(header_file)

                # allocate vars to store data in files and compute the mean
                data_fileslist = np.zeros((num_input_files, num_rows-1, num_cols-1))
            else:
                if header_this != header_file:
                    message = 'header in file: \'%s\' not equal to header found previously: \'%s\'' % (header_this, header_file)
                    CatchErrorException(message)
                if rows1elem_this != rows1elem_file:
                    message = '1st column in file: \'%s\' not equal to 1st column found previously: \'%s\'' % (rows1elem_this, rows1elem_file)
                    CatchErrorException(message)

            # store data corresponding to this file
            data_fileslist[i, :, :] = data_this
        # endfor


        # Compute mean of data along the first dimension of array (input files)
        mean_data_fileslist = np.mean(data_fileslist, axis=0)


        print("Save mean results in file: \'%s\'..." % (args.outputfile))
        fout = open(args.outputfile, 'w')

        strheader = ', '.join(header_file) + '\n'
        fout.write(strheader)

        for i in range (num_rows-1):
            data_thisrow = mean_data_fileslist[i]
            strdata = ', '.join([rows1elem_file[i+1]] + ['%0.6f'%(elem) for elem in data_thisrow]) + '\n'
            fout.write(strdata)
        # endfor

        fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfiles', type=str, nargs='*')
    parser.add_argument('--outputfile', type=str, default='results_mean.txt')
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