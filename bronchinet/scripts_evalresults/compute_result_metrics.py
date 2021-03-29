
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
    #pattern_search_input_files      = get_pattern_prefix_filename(list(indict_reference_keys.values())[0])
    pattern_search_input_files      = 'Sujeto[0-9][0-9]-[a-z]+_'

    list_metrics = OrderedDict()
    for itype_metric in args.list_type_metrics:
        new_metric = get_metric(itype_metric)
        list_metrics[new_metric._name_fun_out] = new_metric
    # endfor



    outdict_calc_metrics = OrderedDict()

    for i, in_predicted_mask_file in enumerate(list_input_predicted_masks_files):
        print("\nInput: \'%s\'..." % (basename(in_predicted_mask_file)))

        in_reference_mask_file = find_file_inlist_same_prefix(basename(in_predicted_mask_file), list_input_reference_masks_files,
                                                              pattern_prefix=pattern_search_input_files)
        print("Reference mask file: \'%s\'..." % (basename(in_reference_mask_file)))

        in_predicted_mask = ImageFileReader.get_image(in_predicted_mask_file)
        in_reference_mask = ImageFileReader.get_image(in_reference_mask_file)
        print("Predictions of size: %s..." % (str(in_predicted_mask.shape)))


        # Compute and store Metrics
        print("\nCompute the Metrics:")
        casename = get_substring_filename(basename(in_predicted_mask_file), substr_pattern=pattern_search_input_files)
        outdict_calc_metrics[casename] = []

        for (imetric_name, imetric) in list_metrics.items():
            outval_metric = imetric.compute(in_reference_mask, in_predicted_mask)

            print("\'%s\': %s..." % (imetric_name, outval_metric))
            outdict_calc_metrics[casename].append(outval_metric)
        # endfor
    # endfor


    # write out computed metrics in file
    fout = open(args.output_file, 'w')
    strheader = ', '.join(['/case/'] + ['/%s/' % (key) for key in list_metrics.keys()]) + '\n'
    fout.write(strheader)

    for (in_casename, outlist_calc_metrics) in outdict_calc_metrics.items():
        list_write_data = [in_casename] + ['%0.6f' % (elem) for elem in outlist_calc_metrics]
        strdata = ', '.join(list_write_data) + '\n'
        fout.write(strdata)
    # endfor
    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('input_predicted_masks_dir', type=str)
    parser.add_argument('input_pred_centrelines_dir', type=str)
    parser.add_argument('--output_file', type=str, default=NAME_PRED_RESULT_METRICS_FILE)
    parser.add_argument('--list_type_metrics', nargs='+', type=str, default=LIST_TYPE_METRICS_RESULT)
    parser.add_argument('--is_remove_trachea_calc_metrics', type=str2bool, default=IS_REMOVE_TRACHEA_CALC_METRICS)
    parser.add_argument('--name_input_reference_masks_relpath', type=str, default=NAME_RAW_LABELS_RELPATH)
    parser.add_argument('--name_input_coarse_airways_relpath', type=str, default=NAME_RAW_COARSEAIRWAYS_RELPATH)
    parser.add_argument('--name_input_reference_centrelines_relpath', type=str, default=NAME_RAW_CENTRELINES_RELPATH)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)