
from common.constant import *
from common.functionutil import *
from common.workdirmanager import TrainDirManager
import math
import argparse


def search_indexes_input_files_from_reference_keys_in_file(in_readfile: str, list_in_reference_files: List[str]) -> List[int]:
    if not is_exist_file(in_readfile):
        message = 'File to specify (Train / Valid / Test) data \'infileorder\ not found: \'%s\'...' % (in_readfile)
        catch_warning_exception(message)
        return []

    out_indexes_input_files = []
    with open(in_readfile, 'r') as fin:
        for in_reference_file in fin.readlines():
            in_reference_file = in_reference_file.replace('\n','').replace('\r','')

            if in_reference_file in list_in_reference_files:
                index_pos_reference_file = [ind for (ind, it_file) in enumerate(list_in_reference_files) if it_file == in_reference_file]
                out_indexes_input_files += index_pos_reference_file
            else:
                message = '\'%s\' not found in list of Input Reference Keys: \'%s\'...' %(in_reference_file, list_in_reference_files)
                catch_error_exception(message)
    # --------------------------------------
    return out_indexes_input_files


LIST_TYPE_DATA_AVAIL  = ['training', 'testing']
LIST_TYPE_DISTRIBUTE_AVAIL = ['original', 'random', 'orderfile']



def main(args):

    workdir_manager         = TrainDirManager(args.basedir)
    input_images_data_path  = workdir_manager.get_datafile_exist(args.name_input_images_relpath)
    in_reference_keys_file  = workdir_manager.get_datafile_exist(args.name_input_reference_keys_file)
    training_data_path      = workdir_manager.get_pathdir_new(args.name_training_data_relpath)
    validation_data_path    = workdir_manager.get_pathdir_new(args.name_validation_data_relpath)
    testing_data_path       = workdir_manager.get_pathdir_new(args.name_testing_data_relpath)
    list_input_images_files = list_files_dir(input_images_data_path)

    if (args.is_prepare_labels):
        input_labels_data_path  = workdir_manager.get_datafile_exist(args.nameI_input_labels_relpath)
        list_input_labels_files = list_files_dir(input_labels_data_path)
        if (len(list_input_images_files) != len(list_input_labels_files)):
            message = 'num Images \'%s\' and Labels \'%s\' not equal...' %(len(list_input_images_files), len(list_input_labels_files))
            catch_error_exception(message)

    if (args.is_input_extra_labels):
        input_extra_labels_data_path  = workdir_manager.get_datafile_exist(args.name_input_extra_labels_relpath)
        list_input_extra_labels_files = list_files_dir(input_extra_labels_data_path)
        if (len(list_input_images_files) != len(list_input_extra_labels_files)):
            message = 'num Images \'%s\' and Extra Labels \'%s\' not equal...' %(len(list_input_images_files), len(list_input_extra_labels_files))
            catch_error_exception(message)



    # Assign indexes for training / validation / testing data (randomly or with fixed order)
    if args.type_distribute == 'original' or args.type_distribute == 'random':
        sum_prop_data = sum(args.dist_prop_data_train_valid_test)
        if sum_prop_data != 1.0:
            message = 'Sum of props of Training / Validation / Testing data != 1.0 (%s)... Change input param...' %(sum_prop_data)
            catch_error_exception(message)

        num_images_files      = len(list_input_images_files)
        num_files_training   = int(math.ceil(args.dist_prop_data_train_valid_test[0] * num_images_files))
        num_files_validation = int(math.ceil(args.dist_prop_data_train_valid_test[1] * num_images_files))
        num_files_testing    = max(0, num_images_files - num_files_training - num_files_validation)
        print("Num files for Training (%s)/ Validation (%s)/ Testing (%s)..." %(num_files_training, num_files_validation, num_files_testing))

        if args.type_distribute == 'random':
            print("Distribute the Training / Validation / Testing data randomly...")
            indexes_input_files = np.random.choice(range(num_images_files), size=num_images_files, replace=False)
        else:
            indexes_input_files = range(num_images_files)

        indexes_training_files   = indexes_input_files[0:num_files_training]
        indexes_validation_files = indexes_input_files[num_files_training:num_files_training+num_files_validation]
        indexes_testing_files    = indexes_input_files[num_files_training+num_files_validation::]

    elif args.type_distribute == 'orderfile':
        args.infile_valid_order = args.infile_train_order.replace('train', 'valid')
        args.infile_test_order  = args.infile_train_order.replace('train', 'test')

        indict_reference_keys    = read_dictionary(in_reference_keys_file)
        list_in_reference_keys   = indict_reference_keys.values()

        indexes_training_files   = search_indexes_input_files_from_reference_keys_in_file(args.infile_train_order, list_in_reference_keys)
        indexes_validation_files = search_indexes_input_files_from_reference_keys_in_file(args.infile_valid_order, list_in_reference_keys)
        indexes_testing_files    = search_indexes_input_files_from_reference_keys_in_file(args.infile_test_order, list_in_reference_keys)

        # Check whether there are files assigned to more than one set (Training / Validation / Testing)
        intersection_check = find_intersection_3lists(indexes_training_files, indexes_validation_files, indexes_testing_files)
        if intersection_check != []:
            list_intersect_files = [indict_reference_keys[basename(list_input_images_files[ind])] for ind in intersection_check]
            message = 'Found files assigned to more than one set (Training / Validation / Testing): %s...' %(list_intersect_files)
            catch_error_exception(message)



    # TRAINING DATA
    if indexes_training_files == []:
        print("No Files assigned to Training Data:")
    else:
        print("Files assigned to Training Data:")
        for index in indexes_training_files:
            input_image_file = list_input_images_files[index]
            output_image_file = join_path_names(training_data_path, basename(input_image_file))
            print("%s --> %s" % (basename(output_image_file), input_image_file))
            makelink(input_image_file, output_image_file)

            if args.is_prepare_labels:
                input_label_file = list_input_labels_files[index]
                output_label_file = join_path_names(training_data_path, basename(input_label_file))
                makelink(input_label_file, output_label_file)

            if args.is_input_extra_labels:
                input_extralabel_file = list_input_extra_labels_files[index]
                output_extralabel_file = join_path_names(training_data_path, basename(input_extralabel_file))
                makelink(input_extralabel_file, output_extralabel_file)
        # endfor


    # VALIDATION DATA
    if indexes_validation_files == []:
        print("No Files assigned to Validation Data:")
    else:
        print("Files assigned to Validation Data:")
        for index in indexes_validation_files:
            input_image_file = list_input_images_files[index]
            output_image_file = join_path_names(validation_data_path, basename(input_image_file))
            print("%s --> %s" % (basename(output_image_file), input_image_file))
            makelink(input_image_file, output_image_file)

            if args.is_prepare_labels:
                input_label_file = list_input_labels_files[index]
                output_label_file = join_path_names(validation_data_path, basename(input_label_file))
                makelink(input_label_file, output_label_file)

            if args.is_input_extra_labels:
                input_extralabel_file = list_input_extra_labels_files[index]
                output_extralabel_file = join_path_names(validation_data_path, basename(input_extralabel_file))
                makelink(input_extralabel_file, output_extralabel_file)
        # endfor


    # TESTING DATA
    if indexes_testing_files == []:
        print("No Files assigned to Testing Data:")
    else:
        print("Files assigned to Testing Data:")
        for index in indexes_testing_files:
            input_image_file = list_input_images_files[index]
            output_image_file = join_path_names(testing_data_path, basename(input_image_file))
            print("%s --> %s" % (basename(output_image_file), input_image_file))
            makelink(input_image_file, output_image_file)

            if args.is_prepare_labels:
                input_label_file = list_input_labels_files[index]
                output_label_file = join_path_names(testing_data_path, basename(input_label_file))
                makelink(input_label_file, output_label_file)

            if args.is_input_extra_labels:
                input_extralabel_file = list_input_extra_labels_files[index]
                output_extralabel_file = join_path_names(testing_data_path, basename(input_extralabel_file))
                makelink(input_extralabel_file, output_extralabel_file)
        # endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--name_input_images_relpath', type=str, default=NAME_PROC_IMAGES_RELPATH)
    parser.add_argument('--nameI_input_labels_relpath', type=str, default=NAME_PROC_LABELS_RELPATH)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    parser.add_argument('--name_input_extra_labels_relpath', type=str, default=NAME_PROC_EXTRA_LABELS_RELPATH)
    parser.add_argument('--name_training_data_relpath', type=str, default=NAME_TRAININGDATA_RELPATH)
    parser.add_argument('--name_validation_data_relpath', type=str, default=NAME_VALIDATIONDATA_RELPATH)
    parser.add_argument('--name_testing_data_relpath', type=str, default=NAME_TESTINGDATA_RELPATH)
    parser.add_argument('--type_data', type=str, default='training')
    parser.add_argument('--type_distribute', type=str, default='original')
    parser.add_argument('--dist_prop_data_train_valid_test', type=str2tuple_float, default=DIST_PROPDATA_TRAINVALIDTEST)
    parser.add_argument('--infile_train_order', type=str, default=None)
    args = parser.parse_args()

    if args.type_data == 'training':
        print("Distribute Training data: Processed Images and Labels...")
        args.is_prepare_labels     = True
        args.is_input_extra_labels = False

    elif args.type_data == 'testing':
        print("Prepare Testing data: Only Processed Images. Keep raw Images and Labels for testing...")
        args.is_keep_raw_images     = True
        args.is_prepare_labels      = False
        args.is_input_extra_labels  = False
        args.is_prepare_centrelines = True
    else:
        message = 'Input param \'typedata\' = \'%s\' not valid, must be inside: \'%s\'...' % (args.type_data, LIST_TYPE_DATA_AVAIL)
        catch_error_exception(message)

    if args.type_distribute not in LIST_TYPE_DISTRIBUTE_AVAIL:
        message = 'Input param \'type_distributedata\' = \'%s\' not valid, must be inside: \'%s\'...' %(args.type_distribute, LIST_TYPE_DISTRIBUTE_AVAIL)
        catch_error_exception(message)

    if args.type_distribute == 'orderfile' and not args.infile_train_order:
        message = 'Input \'infileorder\' file for \'fixed-order\' data distribution is needed...'
        catch_error_exception(message)

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)