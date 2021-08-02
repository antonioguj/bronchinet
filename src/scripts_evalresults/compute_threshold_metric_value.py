
import numpy as np
from tqdm import tqdm
import argparse

from common.constant import BASEDIR, METRIC_EVALUATE_THRESHOLD, NAME_RAW_LABELS_RELPATH, NAME_RAW_CENTRELINES_RELPATH,\
    IS_REMOVE_TRACHEA_CALC_METRICS, NAME_REFERENCE_KEYS_PROCIMAGE_FILE, NAME_RAW_COARSEAIRWAYS_RELPATH
from common.functionutil import basename, list_files_dir, get_regex_pattern_filename, find_file_inlist_with_pattern, \
    str2bool, str2int, str2float, read_dictionary
from common.exceptionmanager import catch_error_exception
from common.workdirmanager import TrainDirManager
from dataloaders.imagefilereader import ImageFileReader
from models.model_manager import get_metric
from imageoperators.imageoperator import MorphoDilateMask, ThresholdImage, ThinningMask, FirstConnectedRegionMask
from imageoperators.maskoperator import MaskOperator


def main(args):

    # SETTINGS
    metric_eval_threshold = args.metric_eval_threshold
    metric_value_sought = args.metric_value_sought
    num_iter_evaluate_max = args.num_iter_evaluate_max
    rel_error_eval_max = args.rel_error_eval_max
    init_threshold_value = args.init_threshold_value

    ref0_threshold_value = 0.0
    if metric_eval_threshold in ['DiceCoefficient', 'AirwayCompleteness']:
        ref0_metric_value = 0.0
    elif metric_eval_threshold in ['AirwayVolumeLeakage', 'AirwayCentrelineLeakage']:
        ref0_metric_value = 1.0
    else:
        message = 'MetricsEvalThreshold \'%s\' not found...' % (metric_eval_threshold)
        catch_error_exception(message)
        ref0_metric_value = None
    _epsilon = 1.0e-06
    # --------

    workdir_manager = TrainDirManager(args.basedir)
    input_posteriors_path = workdir_manager.get_pathdir_exist(args.input_posteriors_dir)
    input_reference_masks_path = workdir_manager.get_datadir_exist(args.name_input_reference_masks_relpath)
    input_reference_cenlines_path = workdir_manager.get_datadir_exist(args.name_input_reference_centrelines_relpath)
    in_reference_keys_file = workdir_manager.get_datafile_exist(args.name_input_reference_keys_file)

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

    metric_calc_cls = get_metric(metric_eval_threshold)
    num_valid_predict_files = len(list_input_posteriors_files)
    curr_thres_value = init_threshold_value
    old_thres_value = ref0_threshold_value
    old_metric_value = ref0_metric_value
    is_comp_converged = False

    # *****************************************************

    # *****************************************************

    print("Load all predictions where to evaluate the metrics, total of \'%s\' files..." % (num_valid_predict_files))
    list_in_posteriors = []
    list_in_reference_masks = []
    list_in_reference_cenlines = []
    list_in_coarse_airways = []

    with tqdm(total=num_valid_predict_files) as progressbar:
        for i, in_posteriors_file in enumerate(list_input_posteriors_files):

            in_posteriors = ImageFileReader.get_image(in_posteriors_file)
            list_in_posteriors.append(in_posteriors)

            in_reference_mask_file = find_file_inlist_with_pattern(basename(in_posteriors_file),
                                                                   list_input_reference_masks_files,
                                                                   pattern_search=pattern_search_infiles)
            in_reference_cenline_file = find_file_inlist_with_pattern(basename(in_posteriors_file),
                                                                      list_input_reference_cenlines_files,
                                                                      pattern_search=pattern_search_infiles)

            in_reference_mask = ImageFileReader.get_image(in_reference_mask_file)
            in_reference_cenline = ImageFileReader.get_image(in_reference_cenline_file)

            if args.is_remove_trachea_calc_metrics:
                in_coarse_airways_file = find_file_inlist_with_pattern(basename(in_posteriors_file),
                                                                       list_input_coarse_airways_files,
                                                                       pattern_search=pattern_search_infiles)
                in_coarse_airways = ImageFileReader.get_image(in_coarse_airways_file)

                in_coarse_airways = MorphoDilateMask.compute(in_coarse_airways, num_iters=4)
                list_in_coarse_airways.append(in_coarse_airways)

                in_reference_mask = MaskOperator.substract_two_masks(in_reference_mask, in_coarse_airways)
                in_reference_cenline = MaskOperator.substract_two_masks(in_reference_cenline, in_coarse_airways)

            list_in_reference_masks.append(in_reference_mask)
            list_in_reference_cenlines.append(in_reference_cenline)
            progressbar.update(1)
        # endfor

    # ******************************

    for iter in range(num_iter_evaluate_max):
        print("Iteration \'%s\'. Evaluate predictions with threshold \'%s\'..." % (iter, curr_thres_value))

        # Loop over all prediction files and compute the mean metrics over the dataset
        with tqdm(total=num_valid_predict_files) as progressbar:
            sumrun_result_metric = 0.0
            for ipos, (in_posteriors, in_reference_mask) in enumerate(zip(list_in_posteriors, list_in_reference_masks)):

                # Compute the binary masks by thresholding the posteriors
                in_predicted_mask = ThresholdImage.compute(in_posteriors, curr_thres_value)

                if args.is_connected_masks:
                    # Compute the first connected component from the binary masks
                    in_predicted_mask = FirstConnectedRegionMask.compute(in_predicted_mask, connectivity_dim=1)

                if metric_calc_cls._is_airway_metric:
                    in_reference_cenline = list_in_reference_cenlines[ipos]

                    try:
                        in_predicted_cenline = ThinningMask.compute(in_predicted_mask)

                    except Exception:
                        # 'catch' issues when predictions are 'weird' (for extreme threshold values)
                        in_predicted_cenline = np.zeros_like(in_predicted_mask)

                if args.is_remove_trachea_calc_metrics:
                    in_coarse_airways = list_in_coarse_airways[ipos]

                    # Remove the trachea and main bronchi from the binary masks
                    in_predicted_mask = MaskOperator.substract_two_masks(in_predicted_mask, in_coarse_airways)

                    if metric_calc_cls._is_airway_metric:
                        in_predicted_cenline = MaskOperator.substract_two_masks(in_predicted_cenline, in_coarse_airways)

                try:
                    if metric_calc_cls._is_airway_metric:
                        out_metric_value = metric_calc_cls.compute(in_reference_mask, in_predicted_mask,
                                                                   in_reference_cenline, in_predicted_cenline)
                    else:
                        out_metric_value = metric_calc_cls.compute(in_reference_mask, in_predicted_mask)

                except Exception:
                    # 'catch' issues when predictions are 'null' (for extreme threshold values)
                    out_metric_value = 0.0

                sumrun_result_metric += out_metric_value
                progressbar.update(1)
            # endfor

            curr_result_metric = sumrun_result_metric / num_valid_predict_files

        # Compare the mean metrics with the value we sought, compute the relative error,
        # and evaluate whether it's close enough
        curr_rel_error = (curr_result_metric - metric_value_sought) / metric_value_sought

        if abs(curr_rel_error) < rel_error_eval_max:
            print("CONVERGED. Found threshold \'%s\', with result metrics \'%s\', rel. error: \'%s\'..."
                  % (curr_thres_value, curr_result_metric, abs(curr_rel_error)))
            is_comp_converged = True
            break
        else:
            # Update the threshold following a Newton-Raphson formula
            new_threshold = curr_thres_value \
                + (curr_thres_value - old_thres_value) / (curr_result_metric - old_metric_value + _epsilon) \
                * (metric_value_sought - curr_result_metric)
            new_threshold = np.clip(new_threshold, 0.0, 1.0)   # clip new value to bounded limits

            print("Not Converged. Result metrics \'%s\', rel. error: \'%s\'. New threshold \'%s\'..."
                  % (curr_result_metric, curr_rel_error, new_threshold))
            old_thres_value = curr_thres_value
            old_metric_value = curr_result_metric
            curr_thres_value = new_threshold
    # endfor

    if not is_comp_converged:
        print("ERROR. COMPUTATION NOT CONVERGED...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_posteriors_dir', type=str)
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--metric_eval_threshold', type=str, default=METRIC_EVALUATE_THRESHOLD)
    parser.add_argument('--metric_value_sought', type=str2float, default=0.13)
    parser.add_argument('--is_remove_trachea_calc_metrics', type=str2bool, default=IS_REMOVE_TRACHEA_CALC_METRICS)
    parser.add_argument('--is_connected_masks', type=str2bool, default=False)
    parser.add_argument('--num_iter_evaluate_max', type=str2int, default=20)
    parser.add_argument('--rel_error_eval_max', type=str2float, default=1.0e-04)
    parser.add_argument('--init_threshold_value', type=str2float, default=0.5)
    parser.add_argument('--name_input_reference_masks_relpath', type=str, default=NAME_RAW_LABELS_RELPATH)
    parser.add_argument('--name_input_reference_centrelines_relpath', type=str, default=NAME_RAW_CENTRELINES_RELPATH)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    parser.add_argument('--name_input_coarse_airways_relpath', type=str, default=NAME_RAW_COARSEAIRWAYS_RELPATH)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
