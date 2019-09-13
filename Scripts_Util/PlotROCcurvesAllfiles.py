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
    index_field_Xaxis = 2
    index_field_Yaxis = 1
    name_metrics_Xaxis = 'Volume Leakage (%)'
    name_metrics_Yaxis = 'Completeness (%)'
    name_input_files_cases = '*_vol[0-9][0-9].txt'
    name_input_files_mean = '*_mean.txt'
    names_outfiles = 'figure_ROCcurve_All.eps'

    labels = ['Unet-lev3', 'Unet-lev5', 'UnetGNN-RegAdj', 'UnetGNN-DynAdj']
    colors = ['green', 'orange', 'blue', 'red']
    # ---------- SETTINGS ----------


    if args.fromfile:
        if not isExistfile(args.listinputdirs):
            message = "File \'%s\' not found..." %(args.listinputdirs)
            CatchErrorException(message)
        fout = open(args.listinputdirs, 'r')
        list_input_dirs = [infile.replace('\n','') for infile in fout.readlines()]
        print("\'inputdirs\' = %s" % (list_input_dirs))
    else:
        list_input_dirs = [infile.replace('\n','') for infile in args.inputdirs]
    num_data_dirs = len(list_input_dirs)

    print("Dirs to plot (\'%s\')..." %(num_data_dirs))
    for i, idir in enumerate(list_input_dirs):
        print("%s: \'%s\'" %(i+1, idir))
    #endfor

    #labels = ['model_%i'%(i+1) for i in range(num_data_dirs)]


    threshold_list = []
    data_Xaxis_list_cases = []
    data_Yaxis_list_cases = []
    data_Xaxis_list_mean = []
    data_Yaxis_list_mean = []
    for (i, in_dir) in enumerate(list_input_dirs):
        list_input_files_cases = findFilesDirAndCheck(in_dir, name_input_files_cases)
        input_file_mean = findFilesDirAndCheck(in_dir, name_input_files_mean)[0]

        # Load data from files for single cases
        data_Xaxis_list_this = []
        data_Yaxis_list_this = []
        for (j, in_data_file) in enumerate(list_input_files_cases):
            data_this = np.loadtxt(in_data_file, dtype=float, skiprows=1, delimiter=',')
            data_Xaxis = data_this[:, index_field_Xaxis] * 100
            data_Yaxis = data_this[:, index_field_Yaxis] * 100

            # eliminate NaNs and dummy values
            data_Xaxis = np.where(data_Xaxis==-1, 0, data_Xaxis)
            data_Xaxis = np.where(np.isnan(data_Xaxis), 0, data_Xaxis)
            data_Yaxis = np.where(data_Yaxis==-1, 0, data_Yaxis)
            data_Yaxis = np.where(np.isnan(data_Yaxis), 0, data_Yaxis)

            data_Xaxis_list_this.append(data_Xaxis)
            data_Yaxis_list_this.append(data_Yaxis)
        #endfor

        data_Xaxis_list_cases.append(data_Xaxis_list_this)
        data_Yaxis_list_cases.append(data_Yaxis_list_this)


        # Load data from file with mean
        data_this = np.loadtxt(input_file_mean, dtype=float, skiprows=1, delimiter=',')
        thresholds = data_this[:, 0]
        data_Xaxis = data_this[:, index_field_Xaxis] * 100
        data_Yaxis = data_this[:, index_field_Yaxis] * 100

        # eliminate NaNs and dummy values
        data_Xaxis = np.where(data_Xaxis==-1, 0, data_Xaxis)
        data_Xaxis = np.where(np.isnan(data_Xaxis), 0, data_Xaxis)
        data_Yaxis = np.where(data_Yaxis==-1, 0, data_Yaxis)
        data_Yaxis = np.where(np.isnan(data_Yaxis), 0, data_Yaxis)

        threshold_list.append (thresholds)
        data_Xaxis_list_mean.append(data_Xaxis)
        data_Yaxis_list_mean.append(data_Yaxis)
    #endfor


    if num_data_dirs == 1:
        # Plot ROC data for single cases
        num_data_files = len(data_Xaxis_list_cases[0])
        for i in range(num_data_files):
            plt.plot(data_Xaxis_list_cases[0][i], data_Yaxis_list_cases[0][i], 'o-', color='b', alpha=0.3)
        #endfor

        # Plot ROC data for mean
        plt.plot(data_Xaxis_list_mean[0], data_Xaxis_list_mean[0], 'o-', color='b')

        plt.xlabel(name_metrics_Xaxis)
        plt.ylabel(name_metrics_Yaxis)
        plt.show()

    else: #num_data_dirs != 1:
        #cmap = plt.get_cmap('rainbow')
        #colors = [cmap(float(i)/(num_data_dirs-1)) for i in range(num_data_dirs)]

        # Plot ROC data for single cases
        for i in range(num_data_dirs):
            num_data_files = len(data_Xaxis_list_cases[i])
            for j in range(num_data_files):
                plt.plot(data_Xaxis_list_cases[i][j], data_Yaxis_list_cases[i][j], color=colors[i], alpha=0.1)
            # endfor
        #endfor

        # Plot ROC data for mean
        for i in range(num_data_dirs):
            plt.plot(data_Xaxis_list_mean[i], data_Yaxis_list_mean[i], color=colors[i], label=labels[i])
        #endfor

        plt.xticks(plt.xticks()[0])
        plt.yticks(plt.yticks()[0])
        plt.xlabel(name_metrics_Xaxis, size=15)
        plt.ylabel(name_metrics_Yaxis, size=15)
        plt.xticks(plt.xticks()[0], size=15)
        plt.yticks(plt.yticks()[0], size=15)
        plt.xlim([0,100])
        plt.ylim([0,100])
        #plt.xlim([7,17])
        #plt.ylim([65,80])
        plt.legend(loc='best', fontsize=15)
        plt.title('ROC curve', size=25)
        plt.show()
        #plt.savefig(names_outfiles, format='eps', dpi=1000)
        #plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdirs', type=str, nargs='*')
    parser.add_argument('--fromfile', type=bool, default=False)
    parser.add_argument('--listinputdirs', type=str, default='listinputfiles.txt')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))
    main(args)


