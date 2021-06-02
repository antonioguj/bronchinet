
from collections import OrderedDict
import argparse

from common.constant import BASEDIR, LIST_TYPE_METRICS_RESULT, NAME_PRED_RESULT_METRICS_FILE, \
    IS_REMOVE_TRACHEA_CALC_METRICS, NAME_RAW_LABELS_RELPATH, NAME_RAW_CENTRELINES_RELPATH, \
    NAME_REFERENCE_KEYS_PROCIMAGE_FILE, NAME_RAW_COARSEAIRWAYS_RELPATH
from common.functionutil import basename, list_files_dir, get_substring_filename, get_regex_pattern_filename, \
    find_file_inlist_with_pattern, str2bool, str2list_str, read_dictionary
from common.exceptionmanager import catch_error_exception
from common.workdirmanager import TrainDirManager
from dataloaders.imagefilereader import ImageFileReader
from models.model_manager import get_metric
from imageoperators.imageoperator import MorphoDilateMask
from imageoperators.maskoperator import MaskOperator


def main(args):

    workdir_manager = TrainDirManager(args.basedir)
    input_predicted_masks_path = workdir_manager.get_pathdir_exist(args.input_predicted_masks_dir)
    input_predicted_cenlines_path = workdir_manager.get_pathdir_exist(args.input_pred_centrelines_dir)
    input_reference_masks_path = workdir_manager.get_datadir_exist(args.name_input_reference_masks_relpath)
    input_reference_cenlines_path = workdir_manager.get_datadir_exist(args.name_input_reference_centrelines_relpath)
    in_reference_keys_file = workdir_manager.get_datafile_exist(args.name_input_reference_keys_file)

    list_input_predicted_masks_files = list_files_dir(input_predicted_masks_path)
    list_input_predicted_cenlines_files = list_files_dir(input_predicted_cenlines_path)
    list_input_reference_masks_files = list_files_dir(input_reference_masks_path)
    list_input_reference_cenlines_files = list_files_dir(input_reference_cenlines_path)
    indict_reference_keys = read_dictionary(in_reference_keys_file)
    pattern_search_infiles = get_regex_pattern_filename(list(indict_reference_keys.values())[0])

    if args.is_remove_trachea_calc_metrics:
        input_coarse_airways_path = workdir_manager.get_datadir_exist(args.name_input_coarse_airways_relpath)
        list_input_coarse_airways_files = list_files_dir(input_coarse_airways_path)
    else:
        list_input_coarse_airways_files = None

    if len(list_input_predicted_masks_files) != len(list_input_predicted_cenlines_files):
        message = 'Input dirs for predicted masks and centrelines have different number of files...'
        catch_error_exception(message)

    list_metrics = OrderedDict()
    for itype_metric in args.list_type_metrics:
        new_metric = get_metric(itype_metric)
        list_metrics[new_metric._name_fun_out] = new_metric
    # endfor

    # *****************************************************

    # *****************************************************

    outdict_calc_metrics = OrderedDict()

    for i, (in_predicted_mask_file, in_predicted_cenline_file) in \
            enumerate(zip(list_input_predicted_masks_files, list_input_predicted_cenlines_files)):
        print("\nInput: \'%s\'..." % (basename(in_predicted_mask_file)))
        print("And: \'%s\'..." % (basename(in_predicted_cenline_file)))

        in_reference_mask_file = find_file_inlist_with_pattern(basename(in_predicted_mask_file),
                                                               list_input_reference_masks_files,
                                                               pattern_search=pattern_search_infiles)
        in_reference_cenline_file = find_file_inlist_with_pattern(basename(in_predicted_mask_file),
                                                                  list_input_reference_cenlines_files,
                                                                  pattern_search=pattern_search_infiles)
        print("Reference mask file: \'%s\'..." % (basename(in_reference_mask_file)))
        print("Reference centrelines file: \'%s\'..." % (basename(in_reference_cenline_file)))

        in_predicted_mask = ImageFileReader.get_image(in_predicted_mask_file)
        in_predicted_cenline = ImageFileReader.get_image(in_predicted_cenline_file)
        in_reference_mask = ImageFileReader.get_image(in_reference_mask_file)
        in_reference_cenline = ImageFileReader.get_image(in_reference_cenline_file)
        print("Predictions of size: %s..." % (str(in_predicted_mask.shape)))

        if args.is_remove_trachea_calc_metrics:
            print("Remove trachea and main bronchi masks in computed metrics...")
            in_coarse_airways_file = find_file_inlist_with_pattern(basename(in_predicted_mask_file),
                                                                   list_input_coarse_airways_files,
                                                                   pattern_search=pattern_search_infiles)
            print("Coarse Airways mask file: \'%s\'..." % (basename(in_coarse_airways_file)))

            in_coarse_airways = ImageFileReader.get_image(in_coarse_airways_file)

            print("Dilate coarse airways masks 4 levels to remove completely the trachea and main bronchi from "
                  "the predictions and the ground-truth...")
            in_coarse_airways = MorphoDilateMask.compute(in_coarse_airways, num_iters=4)

            in_predicted_mask = MaskOperator.substract_two_masks(in_predicted_mask, in_coarse_airways)
            in_predicted_cenline = MaskOperator.substract_two_masks(in_predicted_cenline, in_coarse_airways)
            in_reference_mask = MaskOperator.substract_two_masks(in_reference_mask, in_coarse_airways)
            in_reference_cenline = MaskOperator.substract_two_masks(in_reference_cenline, in_coarse_airways)

        # ******************************

        # Compute and store Metrics
        print("\nCompute the Metrics:")
        casename = get_substring_filename(basename(in_predicted_mask_file), pattern_search=pattern_search_infiles)
        outdict_calc_metrics[casename] = []

        for (imetric_name, imetric) in list_metrics.items():
            if imetric._is_use_voxelsize:
                in_mask_voxel_size = ImageFileReader.get_image_voxelsize(in_predicted_mask_file)
                imetric.set_voxel_size(in_mask_voxel_size)

            if imetric._is_airway_metric:
                outval_metric = imetric.compute(in_reference_mask, in_predicted_mask,
                                                in_reference_cenline, in_predicted_cenline)
            else:
                outval_metric = imetric.compute(in_reference_mask, in_predicted_mask)

            print("\'%s\': %s..." % (imetric_name, outval_metric))
            outdict_calc_metrics[casename].append(outval_metric)
        # endfor
    # endfor

    # *****************************************************

    # write out computed metrics in file
    with open(args.output_file, 'w') as fout:
        strheader = ', '.join(['/case/'] + ['/%s/' % (key) for key in list_metrics.keys()]) + '\n'
        fout.write(strheader)

        for (in_casename, outlist_calc_metrics) in outdict_calc_metrics.items():
            list_write_data = [in_casename] + ['%0.6f' % (elem) for elem in outlist_calc_metrics]
            strdata = ', '.join(list_write_data) + '\n'
            fout.write(strdata)
        # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_predicted_masks_dir', type=str)
    parser.add_argument('input_pred_centrelines_dir', type=str)
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--list_type_metrics', type=str2list_str, default=LIST_TYPE_METRICS_RESULT)
    parser.add_argument('--output_file', type=str, default=NAME_PRED_RESULT_METRICS_FILE)
    parser.add_argument('--is_remove_trachea_calc_metrics', type=str2bool, default=IS_REMOVE_TRACHEA_CALC_METRICS)
    parser.add_argument('--name_input_reference_masks_relpath', type=str, default=NAME_RAW_LABELS_RELPATH)
    parser.add_argument('--name_input_reference_centrelines_relpath', type=str, default=NAME_RAW_CENTRELINES_RELPATH)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    parser.add_argument('--name_input_coarse_airways_relpath', type=str, default=NAME_RAW_COARSEAIRWAYS_RELPATH)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
