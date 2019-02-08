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

    list_input_data_files = ['SavedPredictions_FromCluster/Predictions_size352x240x120_LossWBEC_SlideWindow/mean_ROCsensTPspecFP.txt',
                             'SavedPredictions_FromCluster/Predictions_size352x240x104_LossDice_SlideWindow/mean_ROCsensTPspecFP.txt',
                             'SavedPredictions_FromCluster/Predictions_size352x240x120_LossWBEC_SlideWindow_TransformImages/mean_ROCsensTPspecFP.txt',
                             'SavedPredictions_FromCluster/Predictions_size352x240x104_LossDice_SlideWindow_TransformImages/mean_ROCsensTPspecFP.txt',
                             'SavedPredictions_FromCluster/Predictions_size352x240x104_LossDice_SlideWindow_ElasticDeformImages/mean_ROCsensTPspecFP.txt']

    num_input_data_files = len(list_input_data_files)

    print("Plot FROC values from %s test files:..." %(num_input_data_files))
    print(', '.join(map(lambda item: '\''+basename(item)+'\'', list_input_data_files)))

    threshold_list = []
    sensitivity_list = []
    FPaverage_list = []
    completeness_list = []
    volumeleakage_list = []
    dice_coeff_list = []
    for (i, in_file) in enumerate(list_input_data_files):
        data_this = np.loadtxt(in_file, skiprows=1)
        data_this[-1,1:] = [0.0, 0.0, 0.0, 0.0, 0.0]

        threshold_list.append(data_this[:, 0])
        sensitivity_list.append(data_this[:, 1])
        FPaverage_list.append(data_this[:, 2])
        completeness_list.append(data_this[:, 3] * 100)
        volumeleakage_list.append(data_this[:, 4] * 100)
        dice_coeff_list.append(data_this[:, 5])
    #endfor


    list_reference1_files = ['SavedPredictions_FromCluster/Predictions_size352x240x120_LossWBEC_SlideWindow/mean_results_leakage_test.txt',
                             'SavedPredictions_FromCluster/Predictions_size352x240x120_LossWBEC_SlideWindow_TransformImages/mean_results_leakage_test.txt',
                             'SavedPredictions_FromCluster/Predictions_size352x240x104_LossDice_SlideWindow/mean_results_leakage_test.txt',
                             'SavedPredictions_FromCluster/Predictions_size352x240x104_LossDice_SlideWindow_TransformImages/mean_results_leakage_test.txt',
                             'SavedPredictions_FromCluster/Predictions_size352x240x104_LossDice_SlideWindow_ElasticDeformImages/mean_results_leakage_test.txt']

    list_reference2_files = ['/home/antonio/Results/AirwaySegmen_LUVAR/Predictions_Adria/best_results_opfronted/mean_leakagetest_outerwall_predictmasks_outerwall_grndtruth.txt']

    completeness_reference1_list = []
    volumeleakage_reference1_list= []

    for (i, in_file) in enumerate(list_reference1_files):
        data_this = np.loadtxt(in_file, skiprows=1, usecols=[1,2])

        completeness_reference1_list.append(data_this[0] * 100)
        volumeleakage_reference1_list.append(data_this[1] * 100)
    #endfor

    completeness_reference2_list = []
    volumeleakage_reference2_list= []

    for (i, in_file) in enumerate(list_reference2_files):
        data_this = np.loadtxt(in_file, skiprows=1, usecols=[1,2])

        completeness_reference2_list.append(data_this[0] * 100)
        volumeleakage_reference2_list.append(data_this[1] * 100)
    #endfor


    labels = ['wBEC_None',
              'dice_None',
              'wBEC_Rigid',
              'dice_Rigid',
              'dice_Elastic']

    if num_input_data_files == 1:
        # plot ROC: sensitivity - specificity
        plt.plot(FPaverage_list[0], sensitivity_list[0], 'o-', color='b')
        # annotate thresholds
        if threshold_list[0] is not None:
            plot_annotations_thresholds(FPaverage_list[0], sensitivity_list[0], threshold_list[0])
        plt.xlabel('FalsePositives Average')
        plt.ylabel('True Positive Rate')
        plt.title('FROC curve')
        plt.show()


        # plot ROC: completeness - volume leakage
        plt.plot(volumeleakage_list[0], completeness_list[0], 'o-', color='b')
        # annotate thresholds
        if threshold_list[0] is not None:
            plot_annotations_thresholds(volumeleakage_list[0], completeness_list[0], threshold_list[0])
        plt.xlabel('Volume Leakage (%)')
        plt.ylabel('Completeness (%)')
        plt.show()


        # plot Dice coefficient - threshold
        plt.plot(threshold_list[0], dice_coeff_list[0], 'o-', color='b')
        plt.xlabel('Threshold')
        plt.ylabel('Dice coefficient')
        plt.show()

    else: #num_input_data_files != 1:
        cmap = plt.get_cmap('rainbow')
        colors = [ cmap(float(i)/(num_input_data_files-1)) for i in range(num_input_data_files) ]

        # plot ROC: sensitivity - specificity
        print("plot ROC: sensitivity - specificity...")
        for i in range(num_input_data_files):
            plt.plot(FPaverage_list[i], sensitivity_list[i], color=colors[i], label=labels[i])
            # find optimal threshold
            index_mark_value = find_index_optimal_threshold_sensitTPspecifFP(FPaverage_list[i], sensitivity_list[i])
            # annotation threshold
            plt.scatter(FPaverage_list[i][index_mark_value], sensitivity_list[i][index_mark_value], marker='o', color=colors[i])

            print("file \'%s\', optimal threshold: \'%s\'..." %(i, threshold_list[i][index_mark_value]))
        # endfor
        plt.xlabel('False Positive Average')
        plt.ylabel('True Positive Rate')
        plt.xlim([0, 400000])
        plt.title('FROC curve')
        plt.legend(loc='right')
        plt.show()


        # plot ROC: completeness - volume leakage
        print("plot ROC: completeness - volume leakage...")
        for i in range(num_input_data_files):
            plt.plot(volumeleakage_list[i], completeness_list[i], color=colors[i], label=labels[i])
            # find optimal threshold
            #index_mark_value = find_index_optimal_threshold_sensitTPspecifFP(volumeleakage_list[i], completeness_list[i])
            # annotation threshold
            #plt.scatter(volumeleakage_list[i][index_mark_value], completeness_list[i][index_mark_value], marker='o', color=colors[i])
            print("file \'%s\', optimal threshold: \'%s\'..." % (i, threshold_list[i][index_mark_value]))
        # endfor
        # # Include annotations of other results
        # if completeness_reference1_list:
        #     plt.scatter(volumeleakage_reference1_list, completeness_reference1_list, marker='^', color='b')
        if completeness_reference2_list:
            plt.scatter(volumeleakage_reference2_list, completeness_reference2_list, s=100, marker='x', color='black', label='reference')
        plt.xlabel('Volume Leakage (%)')
        plt.ylabel('Completeness (%)')
        plt.xlim([0,100])
        plt.ylim([0,100])
        plt.legend(loc='right')
        plt.show()


        # plot Dice coefficient - threshold
        print("plot Dice...")
        for i in range(num_input_data_files):
            plt.plot(threshold_list[i], dice_coeff_list[i], color=colors[i], label=labels[i])
            # find optimal threshold
            index_mark_value = find_index_optimal_threshold_dice_coeff(dice_coeff_list[i])
            print("file \'%s\', optimal threshold: \'%s\'..." % (i, threshold_list[i][index_mark_value]))
        #endfor
        plt.xlabel('Threshold')
        plt.ylabel('Dice coefficient')
        plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))
    main(args)
