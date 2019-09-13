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
    index_field_plot = 2
    name_field_plot = 'Completeness'
    # ---------- SETTINGS ----------


    list_subdirs = findFilesDir(args.inputfilesdir, names_subdirs)
    list_subdirs = sorted(list_subdirs, key=getIntegerInString)

    list_input_files = []
    for isubdir in list_subdirs:
        list_input_files.append(joinpathnames(isubdir, names_inputfiles))
    #endfor
    num_data_files = len(list_input_files)


    labels = [str(getIntegerInString(elem)) for elem in list_subdirs]
    labels_xaxis = []
    for i in range(len(labels)):
        if (i+1)%10==0:
            labels_xaxis.append(str(labels[i]))
        else:
            labels_xaxis.append('')
    #endfor
    print("Found num files to plot: \'%s\'..." %(num_data_files))


    list_data_files = []
    for i, in_data_file in enumerate(list_input_files):
        data_this_float = np.genfromtxt(in_data_file, dtype=float, delimiter=',')
        data_field = data_this_float[1:, index_field_plot]
        list_data_files.append(data_field)
    #endfor


    plt.boxplot(list_data_files, labels=labels)
    plt.xticks(plt.xticks()[0], labels_xaxis, size=15)
    plt.yticks(plt.yticks()[0], size=15)
    plt.title('Train History '+ name_field_plot, size=25)
    plt.show()
    #plt.savefig(names_outfiles[i], format='eps', dpi=1000)
    #plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfilesdir', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)