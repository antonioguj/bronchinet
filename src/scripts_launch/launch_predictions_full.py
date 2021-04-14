
import subprocess
import traceback
import sys
import argparse

from common.constant import BASEDIR, NAME_TESTINGDATA_RELPATH, POST_THRESHOLD_VALUE, LIST_TYPE_METRICS_RESULT, \
    IS_ATTACH_COARSE_AIRWAYS, IS_REMOVE_TRACHEA_CALC_METRICS, IS_MASK_REGION_INTEREST, IS_CROP_IMAGES, \
    IS_RESCALE_IMAGES, NAME_CONFIG_PARAMS_FILE, NAME_RAW_LABELS_RELPATH, NAME_TEMPO_POSTERIORS_RELPATH, \
    NAME_POSTERIORS_RELPATH, NAME_PRED_BINARYMASKS_RELPATH, NAME_PRED_CENTRELINES_RELPATH, NAME_RAW_ROIMASKS_RELPATH, \
    NAME_RAW_COARSEAIRWAYS_RELPATH, NAME_REFERENCE_KEYS_POSTERIORS_FILE, NAME_PRED_RESULT_METRICS_FILE, \
    IS_TWO_BOUNDBOXES_LUNGS
from common.functionutil import currentdir, makedir, set_filename_suffix, set_dirname_suffix, is_exist_file, \
    join_path_names, basename, basenamedir, dirname, list_dirs_dir, str2bool, str2list_str
from common.exceptionmanager import catch_error_exception
from common.workdirmanager import TrainDirManager

CODEDIR = join_path_names(BASEDIR, 'Code/')
SCRIPT_PREDICT_MODEL = join_path_names(CODEDIR, 'scripts_experiments/predict_model.py')
SCRIPT_MERGE_DICTIONARIES_PREDS = join_path_names(CODEDIR, 'scripts_evalresults/merge_pydictionaries_preds.py')
SCRIPT_POSTPROCESS_PREDICTIONS = join_path_names(CODEDIR, 'scripts_evalresults/postprocess_predictions.py')
SCRIPT_PROCESS_PREDICT_AIRWAY_TREE = join_path_names(CODEDIR, 'scripts_evalresults/process_predicted_airway_tree.py')
SCRIPT_CALC_CENTRELINES_FROM_MASK = join_path_names(CODEDIR, 'scripts_util/apply_operation_images.py')
SCRIPT_CALC_FIRSTCONNREGION_FROM_MASK = join_path_names(CODEDIR, 'scripts_util/apply_operation_images.py')
SCRIPT_CALC_CONSER_COARSEAIRWAYS_MASK = join_path_names(CODEDIR, 'scripts_util/apply_operation_images.py')
SCRIPT_COMPUTE_RESULT_METRICS = join_path_names(CODEDIR, 'scripts_evalresults/compute_result_metrics.py')


def print_call(new_call):
    message = ' '.join(new_call)
    print("\n" + "*" * 100)
    print("<<< Launch: %s >>>" % (message))
    print("*" * 100 + "\n")


def launch_call(new_call):
    popen_obj = subprocess.Popen(new_call)
    popen_obj.wait()


def create_task_replace_dirs(input_dir, input_dir_to_replace):
    new_call_1 = ['rm', '-r', input_dir]
    new_call_2 = ['mv', input_dir_to_replace, input_dir]
    return [new_call_1, new_call_2]


def main(args):

    inputdir = dirname(args.input_model_file)
    basedir = currentdir()

    # output_basedir = update_dirname(args.output_basedir)
    output_basedir = args.output_basedir
    makedir(output_basedir)

    name_tempo_posteriors_relpath = basenamedir(NAME_TEMPO_POSTERIORS_RELPATH)
    name_posteriors_relpath = basenamedir(NAME_POSTERIORS_RELPATH)
    name_predict_binary_masks_relpath = basenamedir(NAME_PRED_BINARYMASKS_RELPATH)
    name_predict_centrelines_relpath = basenamedir(NAME_PRED_CENTRELINES_RELPATH)
    name_predict_reference_keys_file = basename(NAME_REFERENCE_KEYS_POSTERIORS_FILE)
    name_output_result_metrics_file = basename(NAME_PRED_RESULT_METRICS_FILE)

    inout_tempo_posteriors_path = join_path_names(output_basedir, name_tempo_posteriors_relpath)
    inout_predict_reference_keys_file = join_path_names(output_basedir, name_predict_reference_keys_file)
    output_posteriors_path = join_path_names(output_basedir, name_posteriors_relpath)
    output_predict_binary_masks_path = join_path_names(output_basedir, name_predict_binary_masks_relpath)
    output_predict_centrelines_path = join_path_names(output_basedir, name_predict_centrelines_relpath)
    output_result_metrics_file = join_path_names(output_basedir, name_output_result_metrics_file)

    # *****************************************************

    list_calls_all = []

    if args.is_preds_crossval:
        input_basedir = dirname(dirname(args.input_model_file))
        input_modeldir = basenamedir(dirname(args.input_model_file))
        input_model_relfile = basename(args.input_model_file)

        in_template_modeldir = '_'.join(input_modeldir.split('_')[:-1]) + '*'
        in_template_testdatadir = basenamedir(args.testing_datadir)[:-1] + '*'

        list_input_modeldirs = list_dirs_dir(input_basedir, in_template_modeldir)
        list_testing_datadirs = list_dirs_dir(basedir, in_template_testdatadir)

        if len(list_input_modeldirs) != len(list_testing_datadirs):
            message = 'For Cross-Val setting: num. input model dirs \'%s\' not equal to num. testing dirs \'%s\'' \
                      % (len(list_input_modeldirs), len(list_testing_datadirs))
            catch_error_exception(message)

        print("\nCompute Predictions from \'%s\' models dirs in a Cross-Val setting..." % (len(list_input_modeldirs)))

        list_predict_reference_keys_files_cvfolds = []

        for i, i_inputdir in enumerate(list_input_modeldirs):
            input_model_file = join_path_names(i_inputdir, input_model_relfile)
            in_config_params_file = join_path_names(i_inputdir, NAME_CONFIG_PARAMS_FILE)
            print("For CV-fold %s: load model file: %s" % (i + 1, input_model_file))

            inout_predict_reference_keys_file_this = \
                set_filename_suffix(inout_predict_reference_keys_file, 'CV%0.2i' % (i + 1))
            list_predict_reference_keys_files_cvfolds.append(inout_predict_reference_keys_file_this)

            if not is_exist_file(in_config_params_file):
                message = "Config params file not found: \'%s\'..." % (in_config_params_file)
                catch_error_exception(message)

            # 1st: Compute model predictions, and posteriors for testing work data
            new_call = ['python3', SCRIPT_PREDICT_MODEL,
                        input_model_file,
                        '--basedir', basedir,
                        '--in_config_file', in_config_params_file,
                        '--testing_datadir', list_testing_datadirs[i],
                        '--name_output_predictions_relpath', inout_tempo_posteriors_path,
                        '--name_output_reference_keys_file', inout_predict_reference_keys_file_this,
                        '--is_backward_compat', str(args.is_backward_compat)]
            list_calls_all.append(new_call)
        # endfor

        # merge all created 'predict_reference_keys_files' for each cv-fold into one
        new_call = ['python3', SCRIPT_MERGE_DICTIONARIES_PREDS,
                    *list_predict_reference_keys_files_cvfolds, inout_predict_reference_keys_file]
        list_calls_all.append(new_call)

    else:
        in_config_params_file = join_path_names(inputdir, NAME_CONFIG_PARAMS_FILE)

        if not is_exist_file(in_config_params_file):
            message = "Config params file not found: \'%s\'..." % (in_config_params_file)
            catch_error_exception(message)

        # 1st: Compute model predictions, and posteriors for testing work data
        new_call = ['python3', SCRIPT_PREDICT_MODEL,
                    args.input_model_file,
                    '--basedir', basedir,
                    '--in_config_file', in_config_params_file,
                    '--testing_datadir', args.testing_datadir,
                    '--name_output_predictions_relpath', inout_tempo_posteriors_path,
                    '--name_output_reference_keys_file', inout_predict_reference_keys_file,
                    '--is_backward_compat', str(args.is_backward_compat)]
        list_calls_all.append(new_call)

        list_predict_reference_keys_files_cvfolds = None

    # ******************************

    # 2nd: Compute post-processed posteriors from work predictions
    new_call = ['python3', SCRIPT_POSTPROCESS_PREDICTIONS,
                '--basedir', basedir,
                '--is_mask_region_interest', str(args.is_mask_region_interest),
                '--is_crop_images', str(args.is_crop_images),
                '--is_rescale_images', str(args.is_rescale_images),
                '--is_binary_predictions', 'True',
                '--is_two_boundboxes_lungs', str(args.is_two_boundboxes_lungs),
                '--name_input_predictions_relpath', inout_tempo_posteriors_path,
                '--name_output_posteriors_relpath', output_posteriors_path,
                '--name_input_reference_keys_file', inout_predict_reference_keys_file]
    list_calls_all.append(new_call)

    # ******************************

    # 3rd: Compute the predicted binary masks from the posteriors
    new_call = ['python3', SCRIPT_PROCESS_PREDICT_AIRWAY_TREE,
                '--basedir', basedir,
                '--post_threshold_value', str(args.post_threshold_value),
                '--is_attach_coarse_airways', str(args.is_attach_coarse_airways),
                '--name_input_posteriors_relpath', output_posteriors_path,
                '--name_output_binary_masks_relpath', output_predict_binary_masks_path]
    list_calls_all.append(new_call)

    # ******************************

    if args.is_connected_masks:
        output_tempo_predict_binary_masks_path = set_dirname_suffix(output_predict_binary_masks_path, 'Tempo')

        # Compute the first connected component from the predicted binary masks
        new_call = ['python3', SCRIPT_CALC_FIRSTCONNREGION_FROM_MASK,
                    output_predict_binary_masks_path, output_tempo_predict_binary_masks_path,
                    '--type', 'firstconreg',
                    '--in_conreg_dim', str(args.in_connregions_dim)]
        list_calls_all.append(new_call)

        new_sublist_calls = create_task_replace_dirs(output_predict_binary_masks_path,
                                                     output_tempo_predict_binary_masks_path)
        list_calls_all += new_sublist_calls

    # ******************************

    # 4th: Compute centrelines by thinning the binary masks
    new_call = ['python3', SCRIPT_CALC_CENTRELINES_FROM_MASK,
                output_predict_binary_masks_path, output_predict_centrelines_path,
                '--type', 'thinning']
    list_calls_all.append(new_call)

    # ******************************

    if len(args.list_type_metrics_result) > 0:
        if args.is_conservative_remove_trachea_calc_metrics:
            # get a conversative coarse airways masks to remove trachea and main bronchi in the computation of metrics

            workdir_manager = TrainDirManager(basedir)
            input_reference_masks_path = workdir_manager.get_datadir_exist(NAME_RAW_LABELS_RELPATH)
            input_roimasks_path = workdir_manager.get_datadir_exist(NAME_RAW_ROIMASKS_RELPATH)
            input_coarse_airways_path = workdir_manager.get_datadir_exist(NAME_RAW_COARSEAIRWAYS_RELPATH)

            output_tempo_refer_coarse_airways_path = workdir_manager.get_datadir_new('CoarseAirways_ReferMasks')
            output_conser_coarse_airways_path = workdir_manager.get_datadir_new('CoarseAirways_Conservative')

            # Mask reference airways masks with the lung mask to compute reference trachea and main bronchi
            new_call = ['python3', SCRIPT_CALC_CONSER_COARSEAIRWAYS_MASK,
                        input_reference_masks_path, output_tempo_refer_coarse_airways_path,
                        '--type', 'substract',
                        '--in_2ndmask_dir', input_roimasks_path]
            list_calls_all.append(new_call)

            # Merge original coarse airways with those coming from the reference masks
            new_call = ['python3', SCRIPT_CALC_CONSER_COARSEAIRWAYS_MASK,
                        input_coarse_airways_path,
                        output_conser_coarse_airways_path,
                        '--type', 'merge',
                        '--in_2ndmask_dir', output_tempo_refer_coarse_airways_path]
            list_calls_all.append(new_call)

            new_call = ['rm', '-r', output_tempo_refer_coarse_airways_path]
            list_calls_all.append(new_call)

            name_input_coarse_airways_relpath = basenamedir(output_conser_coarse_airways_path)
        else:
            name_input_coarse_airways_relpath = NAME_RAW_COARSEAIRWAYS_RELPATH
            output_conser_coarse_airways_path = None

        # ******************************

        # 5th: Compute testing metrics from predicted binary masks and centrelines
        new_call = ['python3', SCRIPT_COMPUTE_RESULT_METRICS,
                    output_predict_binary_masks_path, output_predict_centrelines_path,
                    '--basedir', basedir,
                    '--list_type_metrics', str(args.list_type_metrics_result),
                    '--output_file', output_result_metrics_file,
                    '--is_remove_trachea_calc_metrics', str(args.is_remove_trachea_calc_metrics),
                    '--name_input_coarse_airways_relpath', name_input_coarse_airways_relpath]
        list_calls_all.append(new_call)

        # ******************************

        if args.is_conservative_remove_trachea_calc_metrics:
            new_call = ['rm', '-r', output_conser_coarse_airways_path]
            list_calls_all.append(new_call)

    # ******************************

    # Remove temporary data for posteriors not needed
    new_call = ['rm', '-r', inout_tempo_posteriors_path]
    list_calls_all.append(new_call)
    new_call = ['rm', inout_predict_reference_keys_file, inout_predict_reference_keys_file.replace('.npy', '.csv')]
    list_calls_all.append(new_call)

    if args.is_preds_crossval:
        list_predict_reference_keys_files_cvfolds_csvs = \
            [elem.replace('.npy', '.csv') for elem in list_predict_reference_keys_files_cvfolds]
        new_call = ['rm', *list_predict_reference_keys_files_cvfolds, *list_predict_reference_keys_files_cvfolds_csvs]
        list_calls_all.append(new_call)

    # ******************************

    # Iterate over the list and carry out call serially
    for icall in list_calls_all:
        print_call(icall)
        try:
            launch_call(icall)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            message = 'Call failed. Stop pipeline...'
            catch_error_exception(message)
        print("\n")
    # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model_file', type=str)
    parser.add_argument('output_basedir', type=str)
    parser.add_argument('--testing_datadir', type=str, default=NAME_TESTINGDATA_RELPATH)
    parser.add_argument('--is_preds_crossval', type=str2bool, default=False)
    parser.add_argument('--is_mask_region_interest', type=str2bool, default=IS_MASK_REGION_INTEREST)
    parser.add_argument('--is_crop_images', type=str2bool, default=IS_CROP_IMAGES)
    parser.add_argument('--is_rescale_images', type=str2bool, default=IS_RESCALE_IMAGES)
    parser.add_argument('--is_two_boundboxes_lungs', type=str2bool, default=IS_TWO_BOUNDBOXES_LUNGS)
    parser.add_argument('--post_threshold_value', type=float, default=POST_THRESHOLD_VALUE)
    parser.add_argument('--is_attach_coarse_airways', type=str2bool, default=IS_ATTACH_COARSE_AIRWAYS)
    parser.add_argument('--is_connected_masks', type=str2bool, default=False)
    parser.add_argument('--in_connregions_dim', type=int, default=3)
    parser.add_argument('--list_type_metrics_result', type=str2list_str, default=LIST_TYPE_METRICS_RESULT)
    parser.add_argument('--is_remove_trachea_calc_metrics', type=str2bool, default=IS_REMOVE_TRACHEA_CALC_METRICS)
    parser.add_argument('--is_conservative_remove_trachea_calc_metrics', type=str2bool, default=True)
    parser.add_argument('--is_backward_compat', type=str2bool, default=False)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
