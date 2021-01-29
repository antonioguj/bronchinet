
from common.constant import *
from common.functionutil import *
from common.workdirmanager import TrainDirManager
from dataloaders.imagefilereader import ImageFileReader
from models.model_manager import get_metric
from imageoperators.imageoperator import MorphoDilateMask
from imageoperators.maskoperator import MaskOperator
from collections import OrderedDict
import argparse



def main(args):

    workdir_manager                 = TrainDirManager(args.basedir)
    input_predicted_masks_path      = workdir_manager.get_pathdir_exist(args.input_predicted_masks_dir)
    input_reference_masks_path      = workdir_manager.get_datadir_exist(args.name_input_reference_masks_relpath)
    in_reference_keys_file          = workdir_manager.get_datafile_exist(args.name_input_reference_keys_file)
    list_input_predicted_masks_files= list_files_dir(input_predicted_masks_path)
    list_input_reference_masks_files= list_files_dir(input_reference_masks_path)
    indict_reference_keys           = read_dictionary(in_reference_keys_file)
    pattern_search_input_files      = get_pattern_refer_filename(list(indict_reference_keys.values())[0])

    if (args.is_remove_trachea_calc_metrics):
        input_coarse_airways_path       = workdir_manager.get_datadir_exist(args.name_input_coarse_airways_relpath)
        list_input_coarse_airways_files = list_files_dir(input_coarse_airways_path)


    list_metrics_compute = OrderedDict()
    list_is_use_reference_cenlines = []
    list_is_use_predicted_cenlines = []
    for itype_metric in args.list_type_metrics_result:
        new_metric = get_metric(itype_metric)
        list_metrics_compute[new_metric._name_fun_out] = new_metric.compute

        list_is_use_reference_cenlines.append(new_metric._is_use_ytrue_cenlines)
        list_is_use_predicted_cenlines.append(new_metric._is_use_ypred_cenlines)
    # endfor

    is_load_reference_cenlines_files = any(list_is_use_reference_cenlines)
    is_load_predicted_cenlines_files = any(list_is_use_predicted_cenlines)

    if (is_load_reference_cenlines_files):
        print("Loading Reference Centrelines...")
        input_reference_cenlines_path       = workdir_manager.get_datadir_exist(args.name_input_reference_centrelines_relpath)
        list_input_reference_cenlines_files = list_files_dir(input_reference_cenlines_path)

    if (is_load_predicted_cenlines_files):
        if not args.input_centrelines_dir:
            message = 'Missing input path for the Predicted Centrelines...'
            catch_error_exception(message)

        print("Loading Predicted Centrelines...")
        input_predicted_cenlines_path       = workdir_manager.get_pathdir_exist(args.input_centrelines_dir)
        list_input_predicted_cenlines_files = list_files_dir(input_predicted_cenlines_path)



    outdict_computed_metrics = OrderedDict()

    for i, in_predicted_mask_file in enumerate(list_input_predicted_masks_files):
        print("\nInput: \'%s\'..." % (basename(in_predicted_mask_file)))

        in_reference_mask_file = find_file_inlist_same_prefix(basename(in_predicted_mask_file), list_input_reference_masks_files,
                                                              pattern_prefix=pattern_search_input_files)
        print("Reference mask file: \'%s\'..." % (basename(in_reference_mask_file)))

        in_predicted_mask = ImageFileReader.get_image(in_predicted_mask_file)
        in_reference_mask = ImageFileReader.get_image(in_reference_mask_file)
        print("Predictions of size: %s..." % (str(in_predicted_mask.shape)))

        if (is_load_reference_cenlines_files):
            in_reference_cenline_file = find_file_inlist_same_prefix(basename(in_predicted_mask_file), list_input_reference_cenlines_files,
                                                                     pattern_prefix=pattern_search_input_files)
            print("Reference centrelines file: \'%s\'..." % (basename(in_reference_cenline_file)))
            in_reference_cenline = ImageFileReader.get_image(in_reference_cenline_file)

        if (is_load_predicted_cenlines_files):
            in_predicted_cenline_file = find_file_inlist_same_prefix(basename(in_predicted_mask_file), list_input_predicted_cenlines_files,
                                                                     pattern_prefix=pattern_search_input_files)
            print("Predicted centrelines file: \'%s\'..." % (basename(in_predicted_cenline_file)))
            in_predicted_cenline = ImageFileReader.get_image(in_predicted_cenline_file)


        if (args.is_remove_trachea_calc_metrics):
            print("Remove trachea and main bronchii masks in computed metrics...")
            in_coarse_airways_file = find_file_inlist_same_prefix(basename(in_predicted_mask_file), list_input_coarse_airways_files,
                                                                  pattern_prefix=pattern_search_input_files)
            print("Coarse Airways mask file: \'%s\'..." % (basename(in_coarse_airways_file)))

            in_coarse_airways = ImageFileReader.get_image(in_coarse_airways_file)

            print("Dilate coarse airways masks 4 levels to remove completely trachea and main bronchi from ground-truth...")
            in_coarse_airways = MorphoDilateMask.compute(in_coarse_airways, num_iters=4)

            in_predicted_mask = MaskOperator.substract_two_masks(in_predicted_mask, in_coarse_airways)
            in_reference_mask = MaskOperator.substract_two_masks(in_reference_mask, in_coarse_airways)

            if (is_load_reference_cenlines_files):
                in_reference_cenline = MaskOperator.substract_two_masks(in_reference_cenline, in_coarse_airways)

            if (is_load_predicted_cenlines_files):
                in_predicted_cenline = MaskOperator.substract_two_masks(in_predicted_cenline, in_coarse_airways)


        # *******************************************************************************
        # Compute and store Metrics
        print("\nCompute the Metrics:")
        key_casename = get_substring_filename(basename(in_predicted_mask_file), substr_pattern=pattern_search_input_files)
        outdict_computed_metrics[key_casename] = []

        for j, (imetric_name, imetric_compute) in enumerate(list_metrics_compute.items()):
            if list_is_use_reference_cenlines[j]:
                in_reference_data = in_reference_cenline
            else:
                in_reference_data = in_reference_mask
            if list_is_use_predicted_cenlines[j]:
                in_predicted_data = in_predicted_cenline
            else:
                in_predicted_data = in_predicted_mask

            try:
                #'try' statement to 'catch' issues when predictions are 'null' (for extreme threshold values)
                out_val_metric = imetric_compute(in_reference_data, in_predicted_data)
            except:
                # set dummy value for cases with issues
                out_val_metric = -1.0
            print("\'%s\': %s..." % (imetric_name, out_val_metric))

            outdict_computed_metrics[key_casename].append(out_val_metric)
        # endfor
    # endfor


    # write out computed metrics in file
    fout = open(args.output_file, 'w')
    strheader = ', '.join(['/case/'] + ['/%s/' % (key) for key in list_metrics_compute.keys()]) + '\n'
    fout.write(strheader)

    for (in_casename, outlist_computed_metrics) in outdict_computed_metrics.items():
        list_outdata = [in_casename] + ['%0.6f' % (elem) for elem in outlist_computed_metrics]
        strdata = ', '.join(list_outdata) + '\n'
        fout.write(strdata)
    # endfor
    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('input_predicted_masks_dir', type=str)
    parser.add_argument('--input_centrelines_dir', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=NAME_PRED_RESULT_METRICS_FILE)
    parser.add_argument('--list_type_metrics_result', nargs='+', type=str, default=LIST_TYPE_METRICS_RESULT)
    parser.add_argument('--is_remove_trachea_calc_metrics', type=str2bool, default=IS_REMOVE_TRACHEA_CALC_METRICS)
    parser.add_argument('--name_input_reference_masks_relpath', type=str, default=NAME_RAW_LABELS_RELPATH)
    parser.add_argument('--name_input_coarse_airways_relpath', type=str, default=NAME_RAW_COARSEAIRWAYS_RELPATH)
    parser.add_argument('--name_input_reference_centrelines_relpath', type=str, default=NAME_RAW_CENTRELINES_RELPATH)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    args = parser.parse_args()

    if not args.input_centrelines_dir:
        args.input_centrelines_dir = args.input_predicted_masks_dir

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)
