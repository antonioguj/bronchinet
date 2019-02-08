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
    # ---------- SETTINGS ----------
    name_input_files = 'av*_ROCsensTPspecFP.txt'

    input_cases_names = ['av24', 'av25', 'av26', 'av28', 'av41']

    # template search files
    temp_search_input_files = 'av[0-9]*'
    # ---------- SETTINGS ----------


    num_input_cases = len(input_cases_names)

    list_all_files_indir = findFilesDir(args.inputdir, name_input_files)

    # search for the files in the directory to compute mean data from
    list_input_files = []
    for in_file in list_all_files_indir:
        name_prefix_case = findFileWithSamePrefix(basename(in_file), temp_search_input_files)
        if name_prefix_case in input_cases_names:
            list_input_files.append(in_file)
        else:
            continue

    num_input_files = len(list_input_files)

    print("Compute mean of data for cases: \'%s\'..." %(', '.join(map(lambda item: '/'+item+'/', input_cases_names))))
    print("Compute mean of data from \'%s\' files:..." %(num_input_files))
    print(', '.join(map(lambda item: '\''+basename(item)+'\'', list_input_files)))

    out_fullfilename = joinpathnames(args.inputdir, 'mean_' + '_'.join(basename(list_input_files[1]).split('_')[1:]))


    for (i, in_file) in enumerate(list_input_files):
        data_file = str(in_file)
        raw_data_this_string = np.genfromtxt(data_file, dtype=str)
        raw_data_this_float = np.genfromtxt(data_file, dtype=float)

        fields_names_this = [item.replace('/','') for item in raw_data_this_string[0,1:]]
        cases_name_this = raw_data_this_string[0,0].replace('/','')
        cases_data_this = list(raw_data_this_float[1:,0])
        data_this = raw_data_this_float[1:,1:]

        if i==0:
            fields_names = fields_names_this
            cases_name = cases_name_this
            cases_data = cases_data_this
            print("Compute mean of data from fields: \'%s\'..." % (', '.join(map(lambda item: '/' + item + '/', fields_names))))
            num_fields = len(fields_names)
            num_cases = len(cases_data)

            #allocate vars to store data in files and compute the mean
            data_fields_files_list = np.zeros((num_input_files, num_cases, num_fields))
        else:
            if fields_names_this != fields_names:
                message = 'fields found in file \'%s\' do not match those found previously: \'%s\'' %(data_file, fields_names)
                CatchErrorException(message)
            if cases_data_this != cases_data:
                message = 'fields found in file \'%s\' do not match those found previously: \'%s\'' %(data_file, cases_data)
                CatchErrorException(message)

        # store data corresponding to this file
        data_fields_files_list[i,:,:] = data_this
    #endfor


    # Compute mean of values along last dimension of array
    mean_data_fields_files = np.mean(data_fields_files_list, axis=0)


    print("Save in file: \'%s\'..." %(out_fullfilename))
    fout = open(out_fullfilename, 'w')

    strheader = '/%s/'%(cases_name) + ' ' + ' '.join(['/%s/'%(elem) for elem in fields_names]) +'\n'
    fout.write(strheader)

    for (i, f_case) in enumerate(cases_data):
        strdata = str(f_case) + ' ' + ' '.join([str(elem) for elem in mean_data_fields_files[i]]) +'\n'
        fout.write(strdata)
    #endfor

    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str, nargs=1)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)