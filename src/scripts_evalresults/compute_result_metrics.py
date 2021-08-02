
from collections import OrderedDict
import argparse

from common.constant import BASEDIR, LIST_TYPE_METRICS_RESULT, NAME_PRED_RESULT_METRICS_FILE, NAME_RAW_LABELS_RELPATH,\
    NAME_REFERENCE_KEYS_PROCIMAGE_FILE
from common.functionutil import basename, list_files_dir, get_substring_filename, get_regex_pattern_filename, \
    find_file_inlist_with_pattern, str2list_str, read_dictionary
from common.workdirmanager import TrainDirManager
from dataloaders.imagefilereader import ImageFileReader
from models.model_manager import get_metric


def main(args):

    workdir_manager = TrainDirManager(args.basedir)
    input_predicted_masks_path = workdir_manager.get_pathdir_exist(args.input_predicted_masks_dir)
    input_reference_masks_path = workdir_manager.get_datadir_exist(args.name_input_reference_masks_relpath)
    in_reference_keys_file = workdir_manager.get_datafile_exist(args.name_input_reference_keys_file)

    list_input_predicted_masks_files = list_files_dir(input_predicted_masks_path)
    list_input_reference_masks_files = list_files_dir(input_reference_masks_path)
    indict_reference_keys = read_dictionary(in_reference_keys_file)
    pattern_search_infiles = get_regex_pattern_filename(list(indict_reference_keys.values())[0])
    pattern_search_infiles = pattern_search_infiles.replace('left', '[a-z]+').replace('right', '[a-z]+')

    dict_metrics_calc = OrderedDict()
    for itype_metric in args.list_type_metrics:
        new_metric = get_metric(itype_metric)
        dict_metrics_calc[new_metric._name_fun_out] = new_metric
    # endfor

    # *****************************************************

    # *****************************************************

    outdict_metrics_data = OrderedDict()

    for i, in_predicted_mask_file in enumerate(list_input_predicted_masks_files):
        print("\nInput: \'%s\'..." % (basename(in_predicted_mask_file)))

        in_reference_mask_file = find_file_inlist_with_pattern(basename(in_predicted_mask_file),
                                                               list_input_reference_masks_files,
                                                               pattern_search=pattern_search_infiles)
        print("Reference mask file: \'%s\'..." % (basename(in_reference_mask_file)))

        in_predicted_mask = ImageFileReader.get_image(in_predicted_mask_file)
        in_reference_mask = ImageFileReader.get_image(in_reference_mask_file)
        print("Predictions of size: %s..." % (str(in_predicted_mask.shape)))

        # Compute and store Metrics
        print("\nCompute the Metrics:")
        casename = get_substring_filename(basename(in_predicted_mask_file), pattern_search=pattern_search_infiles)
        outdict_metrics_data[casename] = []

        for (imetric_name, imetric_cls) in dict_metrics_calc.items():
            out_metric_value = imetric_cls.compute(in_reference_mask, in_predicted_mask)

            print("\'%s\': %s..." % (imetric_name, out_metric_value))
            outdict_metrics_data[casename].append(out_metric_value)
        # endfor
    # endfor

    # *****************************************************

    # write out computed metrics in file
    with open(args.output_file, 'w') as fout:
        strheader = ', '.join(['/case/'] + ['/%s/' % (key) for key in dict_metrics_calc.keys()]) + '\n'
        fout.write(strheader)

        for (in_casename, outlist_metrics_data) in outdict_metrics_data.items():
            list_write_data = [in_casename] + ['%0.6f' % (elem) for elem in outlist_metrics_data]
            strdata = ', '.join(list_write_data) + '\n'
            fout.write(strdata)
        # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_predicted_masks_dir', type=str)
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--list_type_metrics', type=str2list_str, default=LIST_TYPE_METRICS_RESULT)
    parser.add_argument('--output_file', type=str, default=NAME_PRED_RESULT_METRICS_FILE)
    parser.add_argument('--name_input_reference_masks_relpath', type=str, default=NAME_RAW_LABELS_RELPATH)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
