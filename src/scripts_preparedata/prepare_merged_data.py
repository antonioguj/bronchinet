
from typing import List, Tuple
from collections import OrderedDict
import numpy as np
import argparse

from common.constant import NAME_PROC_IMAGES_RELPATH, NAME_PROC_LABELS_RELPATH, NAME_PROC_EXTRALABELS_RELPATH, \
    NAME_REFERENCE_KEYS_PROCIMAGE_FILE
from common.functionutil import makedir, makelink, copyfile, update_dirname, is_exist_file, is_exist_dir, \
    join_path_names, list_files_dir, basename, basenamedir, basename_filenoext, dirnamedir, str2bool, \
    read_dictionary, save_dictionary, save_dictionary_csv
from common.exceptionmanager import catch_error_exception
from common.workdirmanager import GeneralDirManager


def search_indexes_input_files_from_reference_keys_in_file(in_readfile: str,
                                                           list_in_reference_files_all_data: List[List[str]]
                                                           ) -> List[Tuple[int, int]]:
    if not is_exist_file(in_readfile):
        message = 'File for fixed-order distribution of data \'infile_order\' not found: \'%s\'...' % (in_readfile)
        catch_error_exception(message)

    out_indexes_input_files = []
    with open(in_readfile, 'r') as fin:
        for in_reference_file in fin.readlines():
            in_reference_file = in_reference_file.replace('\n', '').replace('\r', '')

            is_found = False
            for icount_data, it_list_in_reference_keys in enumerate(list_in_reference_files_all_data):
                if in_reference_file in it_list_in_reference_keys:
                    index_pos_reference_file = it_list_in_reference_keys.index(in_reference_file)
                    out_indexes_input_files.append((icount_data, index_pos_reference_file))
                    is_found = True
                    break
            # endfor
            if not is_found:
                list_all_in_reference_files = sum(list_in_reference_files_all_data, [])
                message = '\'%s\' not found in list of Input Reference Keys: \'%s\'...' % \
                          (in_reference_file, list_all_in_reference_files)
                catch_error_exception(message)

    return out_indexes_input_files


LIST_TYPE_DATA_AVAIL = ['training', 'testing']
LIST_TYPE_DISTRIBUTE_AVAIL = ['original', 'random', 'orderfile']


def main(args):

    # SETTINGS
    name_template_output_images_files = 'images_proc-%0.2i.nii.gz'
    name_template_output_labels_files = 'labels_proc-%0.2i.nii.gz'
    name_template_output_extra_labels_files = 'cenlines_proc-%0.2i.nii.gz'

    list_input_images_files_all_data = []
    list_input_labels_files_all_data = []
    list_input_extra_labels_files_all_data = []
    list_dict_in_reference_keys_all_data = []

    for imerge_name_data_path in args.list_merge_data_paths:
        if not is_exist_dir(imerge_name_data_path):
            message = 'Base Data dir: \'%s\' does not exist...' % (imerge_name_data_path)
            catch_error_exception(message)

        workdir_manager = GeneralDirManager(imerge_name_data_path)
        input_images_path = workdir_manager.get_pathdir_exist(args.name_inout_images_relpath)
        in_reference_keys_file = workdir_manager.get_pathfile_exist(args.name_inout_reference_keys_file)

        list_input_images_files = list_files_dir(input_images_path)
        indict_reference_keys = read_dictionary(in_reference_keys_file)
        list_input_images_files_all_data.append(list_input_images_files)
        list_dict_in_reference_keys_all_data.append(indict_reference_keys)

        if args.is_prepare_labels:
            input_labels_path = workdir_manager.get_pathdir_exist(args.name_inout_labels_relpath)
            list_input_labels_files = list_files_dir(input_labels_path)
            list_input_labels_files_all_data.append(list_input_labels_files)

        if args.is_input_extra_labels:
            input_extra_labels_path = workdir_manager.get_pathdir_exist(args.name_inout_extra_labels_relpath)
            list_input_extra_labels_files = list_files_dir(input_extra_labels_path)
            list_input_extra_labels_files_all_data.append(list_input_extra_labels_files)
    # endfor

    # Assign indexes for merging the source data (randomly or with fixed order)
    if args.type_distribute == 'original' or args.type_distribute == 'random':
        indexes_merge_input_files = []

        for idata, it_list_input_images_files in enumerate(list_input_images_files_all_data):
            indexes_files_this = [(idata, index_file) for index_file in range(len(it_list_input_images_files))]
            indexes_merge_input_files += indexes_files_this
        # endfor

        if args.type_distribute == 'random':
            print("Randomly shuffle the merged data...")
            np.random.shuffle(indexes_merge_input_files)

    elif args.type_distribute == 'orderfile':
        list_in_reference_keys_all_data = [list(elem.values()) for elem in list_dict_in_reference_keys_all_data]
        indexes_merge_input_files = \
            search_indexes_input_files_from_reference_keys_in_file(args.infile_order, list_in_reference_keys_all_data)

    else:
        indexes_merge_input_files = None

    # *****************************************************

    # Create new base dir with merged data
    homedir = dirnamedir(args.list_merge_data_paths[0])
    datadir = '-'.join(basenamedir(idir).split('_')[0] for idir in args.list_merge_data_paths) + '_Processed'
    datadir = join_path_names(homedir, datadir)

    output_datadir = update_dirname(datadir)
    makedir(output_datadir)
    # output_datadir = data_dir

    workdir_manager = GeneralDirManager(output_datadir)
    output_images_data_path = workdir_manager.get_pathdir_new(args.name_inout_images_relpath)
    out_reference_keys_file = workdir_manager.get_pathfile_new(args.name_inout_reference_keys_file)

    if args.is_prepare_labels:
        output_labels_data_path = workdir_manager.get_pathdir_new(args.name_inout_labels_relpath)
    else:
        output_labels_data_path = None

    if args.is_input_extra_labels:
        output_extra_labels_data_path = workdir_manager.get_pathdir_new(args.name_inout_extra_labels_relpath)
    else:
        output_extra_labels_data_path = None

    # *****************************************************

    outdict_reference_keys = OrderedDict()

    for icount, (index_data, index_image_file) in enumerate(indexes_merge_input_files):

        input_image_file = list_input_images_files_all_data[index_data][index_image_file]
        in_reference_file = list_dict_in_reference_keys_all_data[index_data][basename_filenoext(input_image_file)]
        output_image_file = join_path_names(output_images_data_path,
                                            name_template_output_images_files % (icount + 1))
        print("%s --> %s (%s)" % (basename(output_image_file), input_image_file, basename(in_reference_file)))
        if args.is_link_merged_files:
            makelink(input_image_file, output_image_file)
        else:
            copyfile(input_image_file, output_image_file)

        outdict_reference_keys[basename_filenoext(output_image_file)] = basename(in_reference_file)

        if args.is_prepare_labels:
            input_label_file = list_input_labels_files_all_data[index_data][index_image_file]
            output_label_file = join_path_names(output_labels_data_path,
                                                name_template_output_labels_files % (icount + 1))
            if args.is_link_merged_files:
                makelink(input_label_file, output_label_file)
            else:
                copyfile(input_label_file, output_label_file)

        if args.is_input_extra_labels:
            input_extra_label_file = list_input_extra_labels_files_all_data[index_data][index_image_file]
            output_extra_label_file = join_path_names(output_extra_labels_data_path,
                                                      name_template_output_extra_labels_files % (icount + 1))
            if args.is_link_merged_files:
                makelink(input_extra_label_file, output_extra_label_file)
            else:
                copyfile(input_extra_label_file, output_extra_label_file)
    # endfor

    # Save reference keys for merged data
    save_dictionary(out_reference_keys_file, outdict_reference_keys)
    save_dictionary_csv(out_reference_keys_file.replace('.npy', '.csv'), outdict_reference_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('list_merge_data_paths', nargs='+', type=str, default=None)
    parser.add_argument('--type_data', type=str, default='training')
    parser.add_argument('--type_distribute', type=str, default='original')
    parser.add_argument('--infile_order', type=str, default=None)
    parser.add_argument('--is_link_merged_files', type=str2bool, default=True)
    parser.add_argument('--name_inout_images_relpath', type=str, default=NAME_PROC_IMAGES_RELPATH)
    parser.add_argument('--name_inout_labels_relpath', type=str, default=NAME_PROC_LABELS_RELPATH)
    parser.add_argument('--name_inout_extra_labels_relpath', type=str, default=NAME_PROC_EXTRALABELS_RELPATH)
    parser.add_argument('--name_inout_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    args = parser.parse_args()

    if args.type_data == 'training':
        print("Distribute Training data: Processed Images and Labels...")
        args.is_prepare_labels = True
        args.is_input_extra_labels = False

    elif args.type_data == 'testing':
        print("Prepare Testing data: Only Processed Images...")
        args.is_prepare_labels = False
        args.is_input_extra_labels = False
    else:
        message = 'Input param \'type_data\' = \'%s\' not valid, must be inside: \'%s\'...' \
                  % (args.type, LIST_TYPE_DATA_AVAIL)
        catch_error_exception(message)

    if args.type_distribute not in LIST_TYPE_DISTRIBUTE_AVAIL:
        message = 'Input param \'type_distribute\' = \'%s\' not valid, must be inside: \'%s\'...' \
                  % (args.type_distribute, LIST_TYPE_DISTRIBUTE_AVAIL)
        catch_error_exception(message)

    if args.type_distribute == 'orderfile' and not args.infile_order:
        message = 'Input \'infile_order\' file for \'orderfile\' data distribution is needed...'
        catch_error_exception(message)

    print("Print input arguments...")
    for key, value in sorted(vars(args).items()):
        print("\'%s\' = %s" % (key, value))

    main(args)
