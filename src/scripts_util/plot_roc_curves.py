
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import argparse

from common.functionutil import is_exist_file
from common.exceptionmanager import catch_error_exception


def plot_annotations_thresholds(xdata: np.ndarray, ydata: np.ndarray, threshold_vals: List[float]) -> None:
    # round thresholds
    threshold_labels = ['%.2f' % elem for elem in threshold_vals]
    xy_buffer = None
    for i, xy in enumerate(zip(xdata, ydata)):
        if xy != xy_buffer:
            plt.annotate(str(threshold_labels[i]), xy=xy, textcoords='data')
            xy_buffer = xy


def find_index_optimal_threshold_tp_fp(xdata: np.ndarray, ydata: np.ndarray) -> int:
    # find the optimal value corresponding to the closest to the left-upper corner.
    index_optim_thres = -1,
    min_dist = 1.0e+06
    max_x = 3.5e+05
    for index_i, (x, y) in enumerate(zip(xdata, ydata)):
        dist = np.sqrt((x / max_x) ** 2 + (y - 1) ** 2)
        if dist < min_dist:
            index_optim_thres = index_i
            min_dist = dist
    return index_optim_thres


def find_index_optimal_threshold_dice_coeff(dice_data: np.ndarray) -> np.ndarray:
    return np.argmax(dice_data)


def main(args):

    # SETTINGS
    index_field_xaxis = 3
    index_field_yaxis = 2
    name_metrics_xaxis = 'Volume Leakage (%)'
    name_metrics_yaxis = 'Completeness (%)'
    # names_outfiles = 'figure_ROCcurve_detail.eps'
    # --------

    if args.fromfile:
        if not is_exist_file(args.list_input_files):
            message = "File \'%s\' not found..." % (args.list_input_files)
            catch_error_exception(message)
        fout = open(args.list_input_files, 'r')
        list_input_files = [infile.replace('\n', '') for infile in fout.readlines()]
        print("\'input_files\' = %s" % (list_input_files))
    else:
        list_input_files = [infile.replace('\n', '') for infile in args.input_files]
    num_input_files = len(list_input_files)

    print("Files to plot ROC curves from: \'%s\'..." % (num_input_files))
    for i, ifile in enumerate(list_input_files):
        print("%s: \'%s\'" % (i + 1, ifile))
    # endfor

    labels_files = ['model_%i' % (i + 1) for i in range(num_input_files)]

    if args.is_annotations:
        if not is_exist_file(args.input_annotate_files):
            message = "Input \'input_annotate_files\' not specified..."
            catch_error_exception(message)
        list_input_annotate_files = [infile.replace('\n', '') for infile in args.input_annotate_files]

        num_annotate_files = len(list_input_annotate_files)
        if num_annotate_files != num_input_files:
            message = "Num annotation files \'%s\' not equal to num data files \'%s\'..." \
                      % (num_annotate_files, num_input_files)
            catch_error_exception(message)

        print("Files for annotations (\'%s\')..." % (num_annotate_files))
        for i, ifile in enumerate(list_input_annotate_files):
            print("%s: \'%s\'" % (i + 1, ifile))
        # endfor
    else:
        num_annotate_files = None
        list_input_annotate_files = None

    # ******************************

    threshold_list = []
    data_xaxis_list = []
    data_yaxis_list = []

    for in_data_file in list_input_files:

        data_this = np.loadtxt(in_data_file, dtype=float, skiprows=1, delimiter=',')
        thresholds = data_this[:, 0]
        data_xaxis = data_this[:, index_field_xaxis] * 100
        data_yaxis = data_this[:, index_field_yaxis] * 100

        # eliminate NaNs and dummy values
        data_xaxis = np.where(data_xaxis == -1, 0, data_xaxis)
        data_xaxis = np.where(np.isnan(data_xaxis), 0, data_xaxis)
        data_yaxis = np.where(data_yaxis == -1, 0, data_yaxis)
        data_yaxis = np.where(np.isnan(data_yaxis), 0, data_yaxis)

        threshold_list.append(thresholds)
        data_xaxis_list.append(data_xaxis)
        data_yaxis_list.append(data_yaxis)
    # endfor

    if args.is_annotations:
        list_annotation_xaxis = []
        list_annotation_yaxis = []

        for in_annot_file in list_input_annotate_files:
            data_this = np.genfromtxt(in_annot_file, dtype=float, delimiter=',')
            data_xaxis = np.mean(data_this[1:, 1 + index_field_xaxis] * 100)
            data_yaxis = np.mean(data_this[1:, 1 + index_field_yaxis] * 100)

            list_annotation_xaxis.append(data_xaxis)
            list_annotation_yaxis.append(data_yaxis)
        # endfor
    else:
        list_annotation_xaxis = None
        list_annotation_yaxis = None

    if num_input_files == 1:
        plt.plot(data_xaxis_list[0], data_yaxis_list[0], 'o-', color='b')

        if args.is_annotations:
            plt.scatter(list_annotation_xaxis[0], list_annotation_yaxis[0], marker='o', color='b', s=50)
        plt.xlabel(name_metrics_xaxis)
        plt.ylabel(name_metrics_yaxis)
        plt.show()

    else:
        # num_input_files != 1:
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(float(i) / (num_input_files - 1)) for i in range(num_input_files)]

        for i in range(num_input_files):
            plt.plot(data_xaxis_list[i], data_yaxis_list[i], color=colors[i], label=labels_files[i])
        # endfor

        if args.is_annotations:
            for i in range(num_annotate_files):
                plt.scatter(list_annotation_xaxis[i], list_annotation_yaxis[i], marker='o', color=colors[i], s=50)
            # endfor

        plt.xticks(plt.xticks()[0])
        plt.yticks(plt.yticks()[0])
        plt.xlabel(name_metrics_xaxis)
        plt.ylabel(name_metrics_yaxis)
        plt.legend(loc='best')
        plt.title('ROC curve')
        plt.show()
        # plt.savefig(names_outfiles, format='eps', dpi=1000)
        # plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', type=str, nargs='*')
    parser.add_argument('--fromfile', type=bool, default=False)
    parser.add_argument('--list_input_files', type=str, default='list_input_files.txt')
    parser.add_argument('--is_annotations', type=bool, default=False)
    parser.add_argument('--input_annotate_files', type=str, nargs='*')
    args = parser.parse_args()

    if args.fromfile and not args.list_input_files:
        message = 'need to input \'list_input_files\' with filenames to plot'
        catch_error_exception(message)

    if args.is_annotations and not args.input_annotate_files:
        message = 'need to input \'input_annotate_files\' with annotation names'
        catch_error_exception(message)

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
