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



def plot_annotations_thresholds(xdata, ydata, threshold_vals):
    # round thresholds
    threshold_labels = ['%.2f' % elem for elem in threshold_vals]
    xy_buffer = None
    for i, xy in enumerate(zip(xdata, ydata)):
        if xy != xy_buffer:
            plt.annotate(str(threshold_labels[i]), xy= xy, textcoords= 'data')
            xy_buffer = xy

def find_index_optimal_threshold_sensitTPspecifFP(xdata, ydata):
    # find the optimal value corresponding to the closest to the left-upper corner.
    index_optim_thres = -1,
    min_dist = 1.0e+06
    max_x = 3.5e+05
    for index_i, (x, y) in enumerate(zip(xdata, ydata)):
        dist = np.sqrt((x / max_x) ** 2 + (y - 1) ** 2)
        if dist < min_dist:
            index_optim_thres = index_i
            min_dist = dist
    # endfor

    return index_optim_thres

def find_index_optimal_threshold_dice_coeff(dice_data):
    return np.argmax(dice_data)



def main(args):
    # ---------- SETTINGS ----------
    index_field_Xaxis = 2
    index_field_Yaxis = 1
    name_metrics_Xaxis = 'Volume Leakage (%)'
    name_metrics_Yaxis = 'Completeness (%)'
    names_outfiles = 'figure_ROCcurve_detail.eps'

    #labels = ['UnetLev3', 'UnetLev5', 'UGnnReg', 'UGnnDyn']
    colors = ['green', 'orange', 'blue', 'red']
    # ---------- SETTINGS ----------


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

    if args.isannotations:
        # if not isExistfile(args.inputannotatefiles):
        #     message = "Input \inputannotatefiles\' not specified..."
        #     CatchErrorException(message)
        # list_input_annotate_files = [infile.replace('\n', '') for infile in args.inputannotatefiles]
        list_input_annotate_files = ['Predictions_ResultsMICCAI-MLMI/AllPredictions_SameLeakage0.13/Predictions_UNet-Lev3_Thres0.1/result_metrics_notrachea.txt',
                                     'Predictions_ResultsMICCAI-MLMI/AllPredictions_SameLeakage0.13/Predictions_UNet-Lev5_Thres0.04/result_metrics_notrachea.txt',
                                     'Predictions_ResultsMICCAI-MLMI/AllPredictions_SameLeakage0.13/Predictions_UNetGNN-RegAdj_Thres0.66/result_metrics_notrachea.txt',
                                     'Predictions_ResultsMICCAI-MLMI/AllPredictions_SameLeakage0.13/Predictions_UNetGNN-DynAdj_Thres0.33/result_metrics_notrachea.txt']
        num_annotate_files = len(list_input_annotate_files)
        if num_annotate_files != num_data_files:
            message = "Num annotation files \'%s\' not equal to num data files \'%s\'..." %(num_annotate_files, num_data_files)
            CatchErrorException(message)

        print("Files for annotations (\'%s\')..." % (num_annotate_files))
        for i, ifile in enumerate(list_input_annotate_files):
            print("%s: \'%s\'" % (i+1, ifile))
        #endfor

    labels = ['model_%i'%(i+1) for i in range(num_data_files)]


    threshold_list = []
    data_Xaxis_list = []
    data_Yaxis_list = []
    for (i, in_file) in enumerate(list_input_files):
        data_this = np.loadtxt(in_file, dtype=float, skiprows=1, delimiter=',')
        thresholds = data_this[:, 0]
        data_Xaxis = data_this[:, index_field_Xaxis] * 100
        data_Yaxis = data_this[:, index_field_Yaxis] * 100

        # eliminate NaNs and dummy values
        data_Xaxis = np.where(data_Xaxis==-1, 0, data_Xaxis)
        data_Xaxis = np.where(np.isnan(data_Xaxis), 0, data_Xaxis)
        data_Yaxis = np.where(data_Yaxis==-1, 0, data_Yaxis)
        data_Yaxis = np.where(np.isnan(data_Yaxis), 0, data_Yaxis)

        threshold_list.append (thresholds)
        data_Xaxis_list.append(data_Xaxis)
        data_Yaxis_list.append(data_Yaxis)
    #endfor

    if args.isannotations:
        annotation_Xaxis_list = []
        annotation_Yaxis_list = []
        for (i, in_file) in enumerate(list_input_annotate_files):
            data_this = np.genfromtxt(in_file, dtype=float, delimiter=',')
            data_Xaxis = np.mean(data_this[1:, 1+index_field_Xaxis] * 100)
            data_Yaxis = np.mean(data_this[1:, 1+index_field_Yaxis] * 100)

            annotation_Xaxis_list.append(data_Xaxis)
            annotation_Yaxis_list.append(data_Yaxis)
        #endfor


    if num_data_files == 1:
        plt.plot(data_Xaxis_list[0], data_Yaxis_list[0], 'o-', color='b')

        if args.isannotations:
            plt.scatter(annotation_Xaxis_list[0], annotation_Yaxis_list[0],  marker='o', color='b', s=50)
        plt.xlabel(name_metrics_Xaxis)
        plt.ylabel(name_metrics_Yaxis)
        plt.show()

    else: #num_data_files != 1:
        #cmap = plt.get_cmap('rainbow')
        #colors = [cmap(float(i)/(num_data_files-1)) for i in range(num_data_files)]

        for i in range(num_data_files):
            plt.plot(data_Xaxis_list[i], data_Yaxis_list[i], color=colors[i], label=labels[i])
        #endfor

        if args.isannotations:
            for i in range(num_annotate_files):
                plt.scatter(annotation_Xaxis_list[i], annotation_Yaxis_list[i], marker='o', color=colors[i], s=50)
            #endfor

        plt.xticks(plt.xticks()[0])
        plt.yticks(plt.yticks()[0])
        plt.xlabel(name_metrics_Xaxis, size=15)
        plt.ylabel(name_metrics_Yaxis, size=15)
        plt.xticks(plt.xticks()[0], size=15)
        plt.yticks(plt.yticks()[0], size=15)
        #plt.xlim([0,100])
        #plt.ylim([0,100])
        #plt.xlim([7,17])
        #plt.ylim([65,80])
        plt.legend(loc='best', fontsize=15)
        plt.title('ROC curve', size=25)
        plt.show()
        #plt.savefig(names_outfiles, format='eps', dpi=1000)
        #plt.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfiles', type=str, nargs='*')
    parser.add_argument('--fromfile', type=bool, default=False)
    parser.add_argument('--listinputfiles', type=str, default='listinputfiles.txt')
    parser.add_argument('--isannotations', type=bool, default=False)
    parser.add_argument('--inputannotatefiles', type=str, nargs='*')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))
    main(args)
