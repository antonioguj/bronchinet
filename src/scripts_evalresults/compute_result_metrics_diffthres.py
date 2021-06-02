
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import argparse

from common.constant import BASEDIR, LIST_TYPE_METRICS_RESULT, IS_REMOVE_TRACHEA_CALC_METRICS, NAME_RAW_LABELS_RELPATH,\
    NAME_RAW_CENTRELINES_RELPATH, NAME_REFERENCE_KEYS_PROCIMAGE_FILE, NAME_RAW_COARSEAIRWAYS_RELPATH
from common.functionutil import is_exist_file, join_path_names, basename, list_files_dir, get_substring_filename, \
    get_regex_pattern_filename, find_file_inlist_with_pattern, str2bool, str2list_str, read_dictionary
from common.workdirmanager import TrainDirManager
from dataloaders.imagefilereader import ImageFileReader
from models.model_manager import get_metric
from imageoperators.imageoperator import MorphoDilateMask, ThresholdImage, ThinningMask, FirstConnectedRegionMask
from imageoperators.maskoperator import MaskOperator


def main(args):

    # SETTINGS
    outfilename_metrics_eachcase = 'results_metrics_diffThres_%s.csv'
    outfilename_metrics_meanall = 'meanAllcases_results_metrics_diffThres.csv'

    # parameters to draw ROC curve
    # inlist_thresholds = [0.0]
    # range_threshold = [-6, -2]
    # inlist_thresholds += (np.logspace(range_threshold[0], range_threshold[1], 5)).tolist()
    # range_threshold = [0.1, 0.5]
    # inlist_thresholds += (np.linspace(range_threshold[0], range_threshold[1], 5)).tolist()
    # inlist_thresholds += [1.0]
    num_thresholds = 11
    range_threshold = [0.0, 1.0]
    inlist_thresholds = [el for el in (np.linspace(range_threshold[0], range_threshold[1], num_thresholds)).tolist()]
    # allowedDistance = 0
    print("List of Threshold values: %s" % (inlist_thresholds))
    # --------

    workdir_manager = TrainDirManager(args.basedir)
    input_posteriors_path = workdir_manager.get_pathdir_exist(args.input_posteriors_dir)
    input_reference_masks_path = workdir_manager.get_datadir_exist(args.name_input_reference_masks_relpath)
    input_reference_cenlines_path = workdir_manager.get_datadir_exist(args.name_input_reference_centrelines_relpath)
    in_reference_keys_file = workdir_manager.get_datafile_exist(args.name_input_reference_keys_file)
    output_files_path = workdir_manager.get_pathdir_new(args.output_dir)

    list_input_posteriors_files = list_files_dir(input_posteriors_path)
    list_input_reference_masks_files = list_files_dir(input_reference_masks_path)
    list_input_reference_cenlines_files = list_files_dir(input_reference_cenlines_path)
    indict_reference_keys = read_dictionary(in_reference_keys_file)
    pattern_search_infiles = get_regex_pattern_filename(list(indict_reference_keys.values())[0])

    if args.is_remove_trachea_calc_metrics:
        input_coarse_airways_path = workdir_manager.get_datadir_exist(args.name_input_coarse_airways_relpath)
        list_input_coarse_airways_files = list_files_dir(input_coarse_airways_path)
    else:
        list_input_coarse_airways_files = None

    list_metrics = OrderedDict()
    for itype_metric in args.list_type_metrics:
        new_metric = get_metric(itype_metric)
        list_metrics[new_metric._name_fun_out] = new_metric
    # endfor

    # *****************************************************

    # *****************************************************

    num_input_files = len(list_input_posteriors_files)
    num_calc_metrics = len(list_metrics)
    num_thresholds = len(inlist_thresholds)
    out_calc_metrics_allcases = np.zeros((num_input_files, num_thresholds, num_calc_metrics))

    for i, in_posteriors_file in enumerate(list_input_posteriors_files):
        print("\nInput: \'%s\'..." % (basename(in_posteriors_file)))

        in_reference_mask_file = find_file_inlist_with_pattern(basename(in_posteriors_file),
                                                               list_input_reference_masks_files,
                                                               pattern_search=pattern_search_infiles)
        in_reference_cenline_file = find_file_inlist_with_pattern(basename(in_posteriors_file),
                                                                  list_input_reference_cenlines_files,
                                                                  pattern_search=pattern_search_infiles)
        print("Reference mask file: \'%s\'..." % (basename(in_reference_mask_file)))
        print("Reference centrelines file: \'%s\'..." % (basename(in_reference_cenline_file)))

        in_posteriors = ImageFileReader.get_image(in_posteriors_file)
        in_reference_mask = ImageFileReader.get_image(in_reference_mask_file)
        in_reference_cenline = ImageFileReader.get_image(in_reference_cenline_file)
        print("Predictions of size: %s..." % (str(in_posteriors.shape)))

        if args.is_remove_trachea_calc_metrics:
            print("Remove trachea and main bronchi masks in computed metrics...")
            in_coarse_airways_file = find_file_inlist_with_pattern(basename(in_posteriors_file),
                                                                   list_input_coarse_airways_files,
                                                                   pattern_search=pattern_search_infiles)
            print("Coarse Airways mask file: \'%s\'..." % (basename(in_coarse_airways_file)))

            in_coarse_airways = ImageFileReader.get_image(in_coarse_airways_file)

            print("Dilate coarse airways masks 4 levels to remove completely the trachea and main bronchi from "
                  "the predictions and the ground-truth...")
            in_coarse_airways = MorphoDilateMask.compute(in_coarse_airways, num_iters=4)

            in_reference_mask = MaskOperator.substract_two_masks(in_reference_mask, in_coarse_airways)
            in_reference_cenline = MaskOperator.substract_two_masks(in_reference_cenline, in_coarse_airways)

        else:
            in_coarse_airways = None
        # ******************************

        # Compute and store Metrics at all thresholds
        print("\nCompute Metrics with all thresholds \'%s\'..." % (inlist_thresholds))
        print("- Compute the binary masks by thresholding the posteriors...")
        if args.is_connected_masks:
            print("- Compute the first connected component from the binary masks...")
        print("- Compute the centrelines by thinning the binary masks...")
        if args.is_remove_trachea_calc_metrics:
            print("- Remove the trachea and main bronchi from the binary masks...")
        print("- Compute the Metrics:")

        casename = get_substring_filename(basename(in_posteriors_file), pattern_search=pattern_search_infiles)
        outdict_calc_metrics = OrderedDict()

        with tqdm(total=num_thresholds) as progressbar:
            for j, in_thres_value in enumerate(inlist_thresholds):

                # Compute the binary masks by thresholding the posteriors
                in_predicted_mask = ThresholdImage.compute(in_posteriors, in_thres_value)

                if args.is_connected_masks:
                    # First, attach the trachea and main bronchi to the binary masks,
                    # to be able to compute the largest connected component
                    in_predicted_mask = MaskOperator.merge_two_masks(in_predicted_mask,
                                                                     in_coarse_airways)  # isNot_intersect_masks=True)

                    # Compute the first connected component from the binary masks
                    in_predicted_mask = FirstConnectedRegionMask.compute(in_predicted_mask, connectivity_dim=1)

                try:
                    # Compute centrelines by thinning the binary masks
                    in_predicted_cenline = ThinningMask.compute(in_predicted_mask)

                except Exception:
                    # 'catch' issues when predictions are 'weird' (for extreme threshold values)
                    in_predicted_cenline = np.zeros_like(in_predicted_mask)

                if args.is_remove_trachea_calc_metrics:
                    # Remove the trachea and main bronchi from the binary masks and centrelines
                    in_predicted_mask = MaskOperator.substract_two_masks(in_predicted_mask, in_coarse_airways)
                    in_predicted_cenline = MaskOperator.substract_two_masks(in_predicted_cenline, in_coarse_airways)

                outdict_calc_metrics[in_thres_value] = []

                for (imetric_name, imetric) in list_metrics.items():
                    if imetric._is_use_voxelsize:
                        in_mask_voxel_size = ImageFileReader.get_image_voxelsize(in_posteriors_file)
                        imetric.set_voxel_size(in_mask_voxel_size)

                    try:
                        if imetric._is_airway_metric:
                            outval_metric = imetric.compute(in_reference_mask, in_predicted_mask,
                                                            in_reference_cenline, in_predicted_cenline)
                        else:
                            outval_metric = imetric.compute(in_reference_mask, in_predicted_mask)

                    except Exception:
                        # 'catch' issues when predictions are 'null' (for extreme threshold values)
                        outval_metric = -1.0

                    outdict_calc_metrics[in_thres_value].append(outval_metric)
                # endfor

                out_calc_metrics_allcases[i, j, :] = outdict_calc_metrics[in_thres_value]
                progressbar.update(1)
            # endfor

        # ******************************

        # write out computed metrics in file
        out_results_filename = outfilename_metrics_eachcase % (casename)
        out_results_filename = join_path_names(output_files_path, out_results_filename)
        if not is_exist_file(out_results_filename):
            with open(out_results_filename, 'w') as fout:
                strheader = ', '.join(['/thres/'] + ['/%s/' % (key) for key in list_metrics.keys()]) + '\n'
                fout.write(strheader)

        with open(out_results_filename, 'a') as fout:
            for (in_thres, outlist_calc_metrics) in outdict_calc_metrics.items():
                list_write_data = ['%0.6f' % (in_thres)] + ['%0.6f' % (elem) for elem in outlist_calc_metrics]
                strdata = ', '.join(list_write_data) + '\n'
                fout.write(strdata)
            # endfor
    # endfor

    # Compute global metrics as mean over all files
    out_mean_allcases_calc_metrics = np.mean(out_calc_metrics_allcases, axis=0)

    out_filename = join_path_names(output_files_path, outfilename_metrics_meanall)
    if not is_exist_file(out_filename):
        with open(out_filename, 'w') as fout:
            strheader = ', '.join(['/thres/'] + ['/%s/' % (key) for key in list_metrics.keys()]) + '\n'
            fout.write(strheader)

    with open(out_filename, 'a') as fout:
        for i, in_thres in enumerate(inlist_thresholds):
            list_write_data = ['%0.6f' % (in_thres)] + ['%0.6f' % (elem) for elem in out_mean_allcases_calc_metrics[i]]
            strdata = ', '.join(list_write_data) + '\n'
            fout.write(strdata)
        # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_posteriors_dir', type=str)
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--list_type_metrics', type=str2list_str, default=LIST_TYPE_METRICS_RESULT)
    parser.add_argument('--is_remove_trachea_calc_metrics', type=str2bool, default=IS_REMOVE_TRACHEA_CALC_METRICS)
    parser.add_argument('--is_connected_masks', type=str2bool, default=False)
    parser.add_argument('--name_input_reference_masks_relpath', type=str, default=NAME_RAW_LABELS_RELPATH)
    parser.add_argument('--name_input_reference_centrelines_relpath', type=str, default=NAME_RAW_CENTRELINES_RELPATH)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    parser.add_argument('--name_input_coarse_airways_relpath', type=str, default=NAME_RAW_COARSEAIRWAYS_RELPATH)
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = args.input_posteriors_dir

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
