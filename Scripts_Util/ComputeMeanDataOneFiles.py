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
    input_cases_names = ['all']
    # ---------- SETTINGS ----------


    print("Compute mean of data from file: \'%s\'..." %(basename(args.inputfile)))

    num_input_cases = len(input_cases_names)

    out_filename = joinpathnames(dirnamepathfile(args.inputfile), 'mean_' + basename(args.inputfile))


    raw_data_string = np.genfromtxt(args.inputfile, dtype=str)
    raw_data_float = np.genfromtxt(args.inputfile, dtype=float)

    fields_names = [item.replace('/','') for item in raw_data_string[0,1:]]
    cases_names = [item.replace('\'','') for item in raw_data_string[1:,0]]
    data = raw_data_float[1:,1:]

    num_fields = len(fields_names)

    indexes_input_cases = []
    if input_cases_names[0] == 'all':
        input_cases_names = []
        for in_case in cases_names:
            input_cases_names.append(in_case)
            indexes_input_cases.append(cases_names.index(in_case))
        #endfor
    else:
        for in_case in input_cases_names:
            if in_case in cases_names:
                indexes_input_cases.append(cases_names.index(in_case))
            else:
                message = 'case \'%s\' not found' %(in_case)
                CatchErrorException(message)

    print("Compute mean of data for fields: \'%s\'..." %(', '.join(map(lambda item: '/'+item+'/', fields_names))))
    print("Compute mean of data for cases: \'%s\'..." %(', '.join(map(lambda item: '/'+item+'/', input_cases_names))))


    # allocate vars to store data in files and compute the mean
    mean_data_fields_files = np.mean(data[indexes_input_cases,:], axis=0)


    print("Save in file: \'%s\'..." %(out_filename))
    fout = open(out_filename, 'w')

    strheader = '/case/ ' + ' '.join(['/%s/'%(elem) for elem in fields_names]) +'\n'
    fout.write(strheader)

    strdata = '\'mean\'' + ' ' + ' '.join([str(elem) for elem in mean_data_fields_files]) + '\n'
    fout.write(strdata)
    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)