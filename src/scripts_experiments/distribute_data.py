
from typing import List
import numpy as np
import math
import argparse

from common.constant import BASEDIR, PROPDATA_TRAIN_VALID_TEST, NAME_PROC_IMAGES_RELPATH, NAME_PROC_LABELS_RELPATH, \
    NAME_TRAININGDATA_RELPATH, NAME_VALIDATIONDATA_RELPATH, NAME_TESTINGDATA_RELPATH, NAME_PROC_EXTRALABELS_RELPATH, \
    NAME_REFERENCE_KEYS_PROCIMAGE_FILE
from common.functionutil import makelink, set_dirname_suffix, is_exist_file, join_path_names, list_files_dir, basename,\
    basename_filenoext, str2int, str2tuple_float, read_dictionary, find_intersection_3lists
from common.exceptionmanager import catch_error_exception, catch_warning_exception
from common.workdirmanager import TrainDirManager


def search_indexes_in_files_from_reference_keys(in_readfile: str, list_in_reference_files: List[str]) -> List[int]:
    if not is_exist_file(in_readfile):
        message = 'File \'infileorder\' to specify Train / Valid / Test data not found: \'%s\'...' % (in_readfile)
        catch_warning_exception(message)

    out_indexes_in_files = []
    with open(in_readfile, 'r') as fin:
        for in_reference_file in fin.readlines():
            in_reference_file = in_reference_file.replace('\n', '').replace('\r', '')
            if in_reference_file in list_in_reference_files:
                index_pos_reference_file = \
                    [ind for (ind, it_file) in enumerate(list_in_reference_files) if it_file == in_reference_file]
                out_indexes_in_files += index_pos_reference_file
            else:
                message = '\'%s\' not found in list of input reference keys: \'%s\'...' \
                          % (in_reference_file, list_in_reference_files)
                catch_error_exception(message)

    return out_indexes_in_files


def check_same_number_files_in_list(list_files_1: List[str], list_files_2: List[str]):
    if len(list_files_1) != len(list_files_2):
        message = 'num files in two lists not equal: \'%s\' != \'%s\'...' % (len(list_files_1), len(list_files_2))
        catch_error_exception(message)


LIST_TYPE_DATA_AVAIL = ['training', 'testing']
LIST_TYPE_DISTRIBUTE_AVAIL = ['original', 'random', 'orderfile', 'crossval', 'crossval_random']


def main(args):

    workdir_manager = TrainDirManager(args.basedir)
    input_images_data_path = workdir_manager.get_datadir_exist(args.name_input_images_relpath)
    in_reference_keys_file = workdir_manager.get_datafile_exist(args.name_input_reference_keys_file)
    list_input_images_files = list_files_dir(input_images_data_path)

    if args.is_prepare_labels:
        input_labels_data_path = workdir_manager.get_datadir_exist(args.name_input_labels_relpath)
        list_input_labels_files = list_files_dir(input_labels_data_path)
        check_same_number_files_in_list(list_input_images_files, list_input_labels_files)
    else:
        list_input_labels_files = None

    if args.is_input_extra_labels:
        input_extra_labels_data_path = workdir_manager.get_datadir_exist(args.name_input_extra_labels_relpath)
        list_input_extra_labels_files = list_files_dir(input_extra_labels_data_path)
        check_same_number_files_in_list(list_input_images_files, list_input_extra_labels_files)
    else:
        list_input_extra_labels_files = None

    # *****************************************************

    # *****************************************************

    # Assign indexes for training / validation / testing data (randomly or with fixed order)
    if args.type_distribute == 'original' or args.type_distribute == 'random':
        sum_prop_data = sum(args.propdata_train_valid_test)
        if sum_prop_data != 1.0:
            message = 'Sum of proportions of Training / Validation / Testing data != 1.0 (%s)... ' \
                      'Change input param...' % (sum_prop_data)
            catch_error_exception(message)

        num_total_files = len(list_input_images_files)
        num_training_files = int(math.ceil(args.propdata_train_valid_test[0] * num_total_files))
        num_validation_files = int(math.ceil(args.propdata_train_valid_test[1] * num_total_files))
        num_testing_files = max(0, num_total_files - num_training_files - num_validation_files)

        print("Num files assigned for Training (%s) / Validation (%s) / Testing (%s)..."
              % (num_training_files, num_validation_files, num_testing_files))

        indexes_input_files = list(range(num_total_files))
        if args.type_distribute == 'random':
            print("Randomly shuffle the data before distributing...")
            np.random.shuffle(indexes_input_files)

        indexes_training_files = indexes_input_files[0:num_training_files]
        indexes_validation_files = indexes_input_files[num_training_files:num_training_files + num_validation_files]
        indexes_testing_files = indexes_input_files[num_training_files + num_validation_files::]

    # ******************************

    elif args.type_distribute == 'orderfile':
        indict_reference_keys = read_dictionary(in_reference_keys_file)
        list_in_reference_keys = list(indict_reference_keys.values())

        indexes_training_files = \
            search_indexes_in_files_from_reference_keys(args.infile_order_train, list_in_reference_keys)
        indexes_validation_files = \
            search_indexes_in_files_from_reference_keys(args.infile_valid_order, list_in_reference_keys)
        indexes_testing_files = \
            search_indexes_in_files_from_reference_keys(args.infile_test_order, list_in_reference_keys)

        # Check whether there are files assigned to more than one set (Training / Validation / Testing)
        intersection_check = \
            find_intersection_3lists(indexes_training_files, indexes_validation_files, indexes_testing_files)
        if intersection_check != []:
            list_intersect_files = \
                [indict_reference_keys[basename_filenoext(list_input_images_files[ind])] for ind in intersection_check]
            message = 'Found files assigned to more than one set (Training / Validation / Testing): %s...' \
                      % (list_intersect_files)
            catch_error_exception(message)

    else:
        indexes_training_files = None
        indexes_validation_files = None
        indexes_testing_files = None

    # *****************************************************

    if args.type_distribute == 'crossval' or args.type_distribute == 'crossval_random':
        cvfolds_info_path = workdir_manager.get_pathdir_new(args.name_cvfolds_info_relpath)
        out_filename_cvfold_info_train = join_path_names(cvfolds_info_path, 'train%0.2i.txt')
        out_filename_cvfold_info_valid = join_path_names(cvfolds_info_path, 'valid%0.2i.txt')
        out_filename_cvfold_info_test = join_path_names(cvfolds_info_path, 'test%0.2i.txt')

        indict_reference_keys = read_dictionary(in_reference_keys_file)

        num_total_files = len(list_input_images_files)
        indexes_input_files = list(range(num_total_files))
        if args.type_distribute == 'crossval_random':
            print("Randomly shuffle the data before distributing, across all cv-folds...")
            np.random.shuffle(indexes_input_files)

        if (num_total_files % args.num_folds_crossval != 0):
            message = "For cross-val, splitting \'%s\' files in \'%s\' folds is not evenly distributed..." \
                      % (num_total_files, args.num_folds_crossval)
            catch_error_exception(message)

        list_indexes_files_split_cvfolds = np.array_split(indexes_input_files, args.num_folds_crossval)
        list_indexes_files_split_cvfolds = [list(elem) for elem in list_indexes_files_split_cvfolds]

        num_testing_files_cvfolds = len(list_indexes_files_split_cvfolds[0])
        num_trainvalid_files_cvfolds = num_total_files - num_testing_files_cvfolds
        prop_valid_in_training_cvfolds = \
            args.propdata_train_valid_test[1] * (num_total_files / (num_trainvalid_files_cvfolds + 1.0e-06))
        num_validation_files_cvfolds = int(math.ceil(prop_valid_in_training_cvfolds * num_trainvalid_files_cvfolds))
        num_training_files_cvfolds = num_trainvalid_files_cvfolds - num_validation_files_cvfolds

        print("Use prop. of valid files in each training cv-fold \'%s\', from input "
              "\'propdata_train_valid_test\'..." % (args.propdata_train_valid_test[1]))
        print("For each cv-fold, num files assigned for Training (%s) / Validation (%s) / Testing (%s)..."
              % (num_training_files_cvfolds, num_validation_files_cvfolds, num_testing_files_cvfolds))

        # ******************************

        # to get ORDERED indexes for training + validation files in cv-folds
        indexes_input_files_repeated = list(np.concatenate((indexes_input_files, indexes_input_files)))

        list_indexes_training_files_cvfolds = []
        list_indexes_validation_files_cvfolds = []
        list_indexes_testing_files_cvfolds = []

        for indexes_files_split_cvfold in list_indexes_files_split_cvfolds:
            pos_last_file_split_in_indexes = indexes_input_files.index(indexes_files_split_cvfold[-1])

            indexes_testing_files = indexes_files_split_cvfold
            indexes_trainvalid_files = \
                indexes_input_files_repeated[pos_last_file_split_in_indexes + 1:
                                             pos_last_file_split_in_indexes + 1 + num_trainvalid_files_cvfolds]
            indexes_training_files = indexes_trainvalid_files[:num_training_files_cvfolds]
            indexes_validation_files = indexes_trainvalid_files[num_training_files_cvfolds:]

            list_indexes_training_files_cvfolds.append(indexes_training_files)
            list_indexes_validation_files_cvfolds.append(indexes_validation_files)
            list_indexes_testing_files_cvfolds.append(indexes_testing_files)
        # endfor

        # ******************************

        def write_file_cvfold_info(in_filename: str, indexes_files: List[int]) -> None:
            with open(in_filename, 'w') as fout:
                for index in indexes_files:
                    in_image_file = list_input_images_files[index]
                    in_reference_key = indict_reference_keys[basename_filenoext(in_image_file)]
                    fout.write('%s\n' % (in_reference_key))

        for i in range(args.num_folds_crossval):
            out_file_cvfold_info_train = out_filename_cvfold_info_train % (i + 1)
            out_file_cvfold_info_valid = out_filename_cvfold_info_valid % (i + 1)
            out_file_cvfold_info_test = out_filename_cvfold_info_test % (i + 1)
            print("For cv-fold %s, write distribution of files for Training / Validation / Testing in: %s, %s, %s..."
                  % (i + 1, basename(out_file_cvfold_info_train), basename(out_file_cvfold_info_valid),
                     basename(out_file_cvfold_info_test)))

            write_file_cvfold_info(out_file_cvfold_info_train, list_indexes_training_files_cvfolds[i])
            write_file_cvfold_info(out_file_cvfold_info_valid, list_indexes_validation_files_cvfolds[i])
            write_file_cvfold_info(out_file_cvfold_info_test, list_indexes_testing_files_cvfolds[i])
        # endfor

    else:
        list_indexes_training_files_cvfolds = None
        list_indexes_validation_files_cvfolds = None
        list_indexes_testing_files_cvfolds = None

    # *****************************************************

    def create_links_images_files_assigned_group(in_list_indexes_files: List[int], out_path_dirname: str) -> None:
        if len(in_list_indexes_files) == 0:
            print("No files assigned...")
        else:
            for index in in_list_indexes_files:
                input_image_file = list_input_images_files[index]
                output_image_file = join_path_names(out_path_dirname, basename(input_image_file))
                print("%s --> %s" % (basename(output_image_file), input_image_file))
                makelink(input_image_file, output_image_file)

                if args.is_prepare_labels:
                    input_label_file = list_input_labels_files[index]
                    output_label_file = join_path_names(out_path_dirname, basename(input_label_file))
                    print("%s --> %s" % (basename(output_label_file), input_label_file))
                    makelink(input_label_file, output_label_file)

                if args.is_input_extra_labels:
                    input_extra_label_file = list_input_extra_labels_files[index]
                    output_extra_label_file = join_path_names(out_path_dirname, basename(input_extra_label_file))
                    makelink(input_extra_label_file, output_extra_label_file)
            # endfor

    if args.type_distribute == 'crossval' or args.type_distribute == 'crossval_random':
        for i in range(args.num_folds_crossval):
            print("\nFor cv-fold %s: Files assigned to Training Data:" % (i + 1))
            training_data_path = workdir_manager.get_pathdir_new(args.name_training_data_relpath % (i + 1))

            create_links_images_files_assigned_group(list_indexes_training_files_cvfolds[i], training_data_path)

            print("For cv-fold %s: Files assigned to Validation Data:" % (i + 1))
            validation_data_path = workdir_manager.get_pathdir_new(args.name_validation_data_relpath % (i + 1))

            create_links_images_files_assigned_group(list_indexes_validation_files_cvfolds[i], validation_data_path)

            print("For cv-fold %s: Files assigned to Testing Data:" % (i + 1))
            testing_data_path = workdir_manager.get_pathdir_new(args.name_testing_data_relpath % (i + 1))

            create_links_images_files_assigned_group(list_indexes_testing_files_cvfolds[i], testing_data_path)
        # endfor
    else:
        print("\nFiles assigned to Training Data:")
        training_data_path = workdir_manager.get_pathdir_new(args.name_training_data_relpath)

        create_links_images_files_assigned_group(indexes_training_files, training_data_path)

        print("Files assigned to Validation Data:")
        validation_data_path = workdir_manager.get_pathdir_new(args.name_validation_data_relpath)

        create_links_images_files_assigned_group(indexes_validation_files, validation_data_path)

        print("Files assigned to Testing Data:")
        testing_data_path = workdir_manager.get_pathdir_new(args.name_testing_data_relpath)

        create_links_images_files_assigned_group(indexes_testing_files, testing_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--type_data', type=str, default='training')
    parser.add_argument('--type_distribute', type=str, default='original')
    parser.add_argument('--propdata_train_valid_test', type=str2tuple_float, default=PROPDATA_TRAIN_VALID_TEST)
    parser.add_argument('--infile_order_train', type=str, default=None)
    parser.add_argument('--num_folds_crossval', type=str2int, default=None)
    parser.add_argument('--name_input_images_relpath', type=str, default=NAME_PROC_IMAGES_RELPATH)
    parser.add_argument('--name_input_labels_relpath', type=str, default=NAME_PROC_LABELS_RELPATH)
    parser.add_argument('--name_training_data_relpath', type=str, default=NAME_TRAININGDATA_RELPATH)
    parser.add_argument('--name_validation_data_relpath', type=str, default=NAME_VALIDATIONDATA_RELPATH)
    parser.add_argument('--name_testing_data_relpath', type=str, default=NAME_TESTINGDATA_RELPATH)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    parser.add_argument('--name_input_extra_labels_relpath', type=str, default=NAME_PROC_EXTRALABELS_RELPATH)
    args = parser.parse_args()

    if args.type_data == 'training':
        print("Distribute Training data: Processed Images and Labels...")
        args.is_prepare_labels = True
        args.is_input_extra_labels = False

    elif args.type_data == 'testing':
        print("Prepare Testing data: Only Processed Images. Keep raw Images and Labels for testing...")
        args.is_keep_raw_images = True
        args.is_prepare_labels = False
        args.is_input_extra_labels = False
        args.is_prepare_centrelines = True
    else:
        message = 'Input param \'type_data\' = \'%s\' not valid, must be inside: \'%s\'...' \
                  % (args.type_data, LIST_TYPE_DATA_AVAIL)
        catch_error_exception(message)

    if args.type_distribute not in LIST_TYPE_DISTRIBUTE_AVAIL:
        message = 'Input param \'type_distribute\' = \'%s\' not valid, must be inside: \'%s\'...' \
                  % (args.type_distribute, LIST_TYPE_DISTRIBUTE_AVAIL)
        catch_error_exception(message)

    if args.type_distribute == 'orderfile':
        if not args.infile_order_train:
            message = 'Input \'infile_order_train\' file for \'orderfile\' data distribution is needed...'
            catch_error_exception(message)
        else:
            args.infile_valid_order = args.infile_order_train.replace('train', 'valid')
            args.infile_test_order = args.infile_order_train.replace('train', 'test')

            if not is_exist_file(args.infile_order_train):
                message = 'Input \'infile_order_train\' file does not exist: \'%s\'...' % (args.infile_order_train)
                catch_error_exception(message)
            if not is_exist_file(args.infile_valid_order):
                message = 'Input \'infile_valid_order\' file does not exist: \'%s\'...' % (args.infile_valid_order)
                catch_error_exception(message)
            if not is_exist_file(args.infile_test_order):
                message = 'Input \'infile_test_order\' file does not exist: \'%s\'...' % (args.infile_test_order)
                catch_error_exception(message)

    if args.type_distribute == 'crossval' or args.type_distribute == 'crossval_random':
        if not args.num_folds_crossval:
            message = 'Input \'num_folds_crossval\' for \'crossval\' data distribution is needed...'
            catch_error_exception(message)
        else:
            args.name_training_data_relpath = set_dirname_suffix(args.name_training_data_relpath, 'CV%0.2i')
            args.name_validation_data_relpath = set_dirname_suffix(args.name_validation_data_relpath, 'CV%0.2i')
            args.name_testing_data_relpath = set_dirname_suffix(args.name_testing_data_relpath, 'CV%0.2i')
            args.name_cvfolds_info_relpath = 'CVfoldsInfo/'

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
