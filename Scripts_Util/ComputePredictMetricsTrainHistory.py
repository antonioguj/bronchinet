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
    # ---------- SETTINGS ----------
    names_subdirs = 'Predictions_e*'
    names_inputfiles = 'result_metrics_notrachea.txt'
    name_outfile  = 'metricsHistory.txt'
    # ---------- SETTINGS ----------


    list_subdirs = findFilesDir(args.inputfilesdir, names_subdirs)
    list_subdirs = sorted(list_subdirs, key=getIntegerInString)

    list_input_files = []
    for isubdir in list_subdirs:
        list_input_files.append(joinpathnames(isubdir, names_inputfiles))
    #endfor
    num_data_files = len(list_input_files)


    out_filename = joinpathnames(args.inputfilesdir, name_outfile)
    print("Write output file: %s..." %(out_filename))
    fout = open(out_filename, 'w')

    for i, in_data_file in enumerate(list_input_files):
        if i==0:
            data_this_string = np.genfromtxt(in_data_file, dtype=str, delimiter=',')
            strheader = '/epoch/,' + ','.join(data_this_string[0,1:]) + '\n'
            fout.write(strheader)

        data_this_float = np.genfromtxt(in_data_file, dtype=float, delimiter=',')
        data_fields = data_this_float[1:, 1:]

        iepoch = getIntegerInString(in_data_file)
        mean_data_fields = np.mean(data_fields, axis=0)

        strdata = ', '.join([str(elem) for elem in [iepoch] + list(mean_data_fields)]) +'\n'
        fout.write(strdata)
    #endfor

    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfilesdir', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)