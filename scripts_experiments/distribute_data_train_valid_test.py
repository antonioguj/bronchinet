
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

def write_file_cvfold_info(in_filename: str, list_in_files) -> None:
    with open(in_filename, 'w') as fout:
        for ifile in list_in_files:
            fout.write('%s\n' %(ifile))

LIST_TYPE_DATA_AVAIL  = ['training', 'testing']
LIST_TYPE_DISTRIBUTE_AVAIL = ['original', 'random', 'orderfile', 'crossval', 'crossval_random']



def main(args):

    workdir_manager         = TrainDirManager(args.basedir)
    input_images_data_path  = workdir_manager.get_datadir_exist(args.name_input_images_relpath)
    in_reference_keys_file  = workdir_manager.get_datafile_exist(args.name_input_reference_keys_file)
    list_input_images_files = list_files_dir(input_images_data_path)

    if (args.is_prepare_labels):
        input_labels_data_path  = workdir_manager.get_datadir_exist(args.name_input_labels_relpath)
        list_input_labels_files = list_files_dir(input_labels_data_path)
        if (len(list_input_images_files) != len(list_input_labels_files)):
            message = 'num Images \'%s\' and Labels \'%s\' not equal...' %(len(list_input_images_files), len(list_input_labels_files))
            catch_error_exception(message)

    if (args.is_input_extra_labels):
        input_extra_labels_data_path  = workdir_manager.get_datadir_exist(args.name_input_extra_labels_relpath)
        list_input_extra_labels_files = list_files_dir(input_extra_labels_data_path)
        if (len(list_input_images_files) != len(list_input_extra_labels_files)):
            message = 'num Images \'%s\' and Extra Labels \'%s\' not equal...' %(len(list_input_images_files), len(list_input_extra_labels_files))
            catch_error_exception(message)



    # Assign indexes for training / validation / testing data (randomly or with fixed order)
    if args.type_distribute == 'original' or args.type_distribute == 'random':
        sum_prop_data = sum(args.dist_propdata_train_valid_test)
        if sum_prop_data != 1.0:
            message = 'Sum of props of Training / Validation / Testing data != 1.0 (%s)... Change input param...' %(sum_prop_data)
            catch_error_exception(message)

        num_total_files      = len(list_input_images_files)
        num_training_files   = int(math.ceil(args.dist_propdata_train_valid_test[0] * num_total_files))
        num_validation_files = int(math.ceil(args.dist_propdata_train_valid_test[1] * num_total_files))
        num_testing_files    = max(0, num_total_files - num_training_files - num_validation_files)

        print("Num files for Training (%s)/ Validation (%s)/ Testing (%s)..." %(num_training_files, num_validation_files, num_testing_files))

        if args.type_distribute == 'random':
            print("Distribute the Training / Validation / Testing data randomly...")
            indexes_input_files = np.random.choice(range(num_total_files), size=num_total_files, replace=False)
        else:
            indexes_input_files = range(num_total_files)

        indexes_training_files   = indexes_input_files[0:num_training_files]
        indexes_validation_files = indexes_input_files[num_training_files:num_training_files + num_validation_files]
        indexes_testing_files    = indexes_input_files[num_training_files + num_validation_files::]

    elif args.type_distribute == 'orderfile':
        indict_reference_keys    = read_dictionary(in_reference_keys_file)
        list_in_reference_keys   = list(indict_reference_keys.values())

        indexes_training_files   = search_indexes_input_files_from_reference_keys_in_file(args.infile_train_order, list_in_reference_keys)
        indexes_validation_files = search_indexes_input_files_from_reference_keys_in_file(args.infile_valid_order, list_in_reference_keys)
        indexes_testing_files    = search_indexes_input_files_from_reference_keys_in_file(args.infile_test_order, list_in_reference_keys)

        # Check whether there are files assigned to more than one set (Training / Validation / Testing)
        intersection_check = find_intersection_3lists(indexes_training_files, indexes_validation_files, indexes_testing_files)
        if intersection_check != []:
            list_intersect_files = [indict_reference_keys[basename_filenoext(list_input_images_files[ind])] for ind in intersection_check]
            message = 'Found files assigned to more than one set (Training / Validation / Testing): %s...' %(list_intersect_files)
            catch_error_exception(message)

    elif args.type_distribute == 'crossval' or \
         args.type_distribute == 'crossval_random':
        cvfolds_info_path = workdir_manager.get_pathdir_new(args.name_cvfolds_info_relpath)
        name_cvfold_info_file_training   = join_path_names(cvfolds_info_path, 'train%0.2i.txt')
        name_cvfold_info_file_validation = join_path_names(cvfolds_info_path, 'valid%0.2i.txt')
        name_cvfold_info_file_testing    = join_path_names(cvfolds_info_path, 'test%0.2i.txt')

        indict_reference_keys = read_dictionary(in_reference_keys_file)

        num_total_files      = len(list_input_images_files)
        if (num_total_files % args.num_folds_crossval != 0):
            message = "Splitting \'%s\' total files in \'%s\' CV-folds does not result in even distribution..." %(num_total_files, args.num_folds_crossval)
            catch_error_exception(message)

        num_testing_files    = num_total_files // args.num_folds_crossval
        num_trainvalid_files = num_total_files - num_testing_files
        propdata_valid_intrainvalid_files = args.dist_propdata_train_valid_test[1] * (num_total_files / (num_trainvalid_files + 1.0e-06))
        num_validation_files = int(math.ceil(propdata_valid_intrainvalid_files * num_trainvalid_files))
        num_training_files   = num_trainvalid_files - num_validation_files

        print("Use proportion \'%s\' of validation files in each cv-fold, from input \'dist_propdata_train_valid_test\'..." %(args.dist_propdata_train_valid_test[1]))
        print("For each CV-folds: num files for Training (%s)/ Validation (%s)/ Testing (%s)..." %(num_training_files, num_validation_files, num_testing_files))

        if args.type_distribute == 'crossval_random':
            print("Before distributing files across cross-validation folds, randomize the Training / Validation / Testing data...")
            indexes_input_files = np.random.choice(range(num_total_files), size=num_total_files, replace=False)
        else:
            indexes_input_files = range(num_total_files)

        list_indexes_cvfolds_training_files   = []
        list_indexes_cvfolds_validation_files = []
        list_indexes_cvfolds_testing_files    = []

        for i in range(args.num_folds_crossval):
            indexes_training_files   = indexes_input_files[0:num_training_files]
            indexes_validation_files = indexes_input_files[num_training_files:num_training_files + num_validation_files]
            indexes_testing_files    = indexes_input_files[num_training_files + num_validation_files::]

            list_indexes_cvfolds_training_files.append(indexes_training_files)
            list_indexes_cvfolds_validation_files.append(indexes_validation_files)
            list_indexes_cvfolds_testing_files.append(indexes_testing_files)

            # save assigned files for training / validation / testing in text files
            list_training_files   = [indict_reference_keys[basename_filenoext(list_input_images_files[ind])] for ind in indexes_training_files]
            list_validation_files = [indict_reference_keys[basename_filenoext(list_input_images_files[ind])] for ind in indexes_validation_files]
            list_testing_files    = [indict_reference_keys[basename_filenoext(list_input_images_files[ind])] for ind in indexes_testing_files]

            write_file_cvfold_info(name_cvfold_info_file_training %(i+1), list_training_files)
            write_file_cvfold_info(name_cvfold_info_file_validation %(i+1), list_validation_files)
            write_file_cvfold_info(name_cvfold_info_file_testing %(i+1), list_testing_files)

            # roll indexes once more to distribute evenly files for next cv-fold
            indexes_input_files = np.roll(indexes_input_files, num_testing_files)
        # endfor



    def create_links_images_files_assigned_group(in_list_indexes_files: List[int], out_name_path: str) -> None:
        if len(in_list_indexes_files) == 0:
            print("No files assigned...")
        else:
            for index in in_list_indexes_files:
                input_image_file = list_input_images_files[index]
                output_image_file = join_path_names(out_name_path, basename(input_image_file))
                print("%s --> %s" % (basename(output_image_file), input_image_file))
                makelink(input_image_file, output_image_file)

                if args.is_prepare_labels:
                    input_label_file = list_input_labels_files[index]
                    output_label_file = join_path_names(out_name_path, basename(input_label_file))
                    makelink(input_label_file, output_label_file)

                if args.is_input_extra_labels:
                    input_extralabel_file = list_input_extra_labels_files[index]
                    output_extralabel_file = join_path_names(out_name_path, basename(input_extralabel_file))
                    makelink(input_extralabel_file, output_extralabel_file)
            # endfor


    if args.type_distribute == 'crossval' or \
       args.type_distribute == 'crossval_random':
        for i in range(args.num_folds_crossval):
            print("\nFor CV-fold %s: Files assigned to Training Data:" %(i+1))
            training_data_path = workdir_manager.get_pathdir_new(args.name_training_data_relpath %(i+1))

            create_links_images_files_assigned_group(list_indexes_cvfolds_training_files[i], training_data_path)

            print("For CV-fold %s: Files assigned to Validation Data:" %(i+1))
            validation_data_path = workdir_manager.get_pathdir_new(args.name_validation_data_relpath %(i+1))

            create_links_images_files_assigned_group(list_indexes_cvfolds_validation_files[i], validation_data_path)

            print("For CV-fold %s: Files assigned to Testing Data:" %(i+1))
            testing_data_path = workdir_manager.get_pathdir_new(args.name_testing_data_relpath %(i+1))

            create_links_images_files_assigned_group(list_indexes_cvfolds_testing_files[i], testing_data_path)
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
    parser.add_argument('--name_input_images_relpath', type=str, default=NAME_PROC_IMAGES_RELPATH)
    parser.add_argument('--name_input_labels_relpath', type=str, default=NAME_PROC_LABELS_RELPATH)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    parser.add_argument('--name_input_extra_labels_relpath', type=str, default=NAME_PROC_EXTRA_LABELS_RELPATH)
    parser.add_argument('--name_training_data_relpath', type=str, default=NAME_TRAININGDATA_RELPATH)
    parser.add_argument('--name_validation_data_relpath', type=str, default=NAME_VALIDATIONDATA_RELPATH)
    parser.add_argument('--name_testing_data_relpath', type=str, default=NAME_TESTINGDATA_RELPATH)
    parser.add_argument('--type_data', type=str, default='training')
    parser.add_argument('--type_distribute', type=str, default='original')
    parser.add_argument('--dist_propdata_train_valid_test', type=str2tuple_float, default=DIST_PROPDATA_TRAINVALIDTEST)
    parser.add_argument('--infile_train_order', type=str, default=None)
    parser.add_argument('--num_folds_crossval', type=int, default=None)
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
        message = 'Input param \'type_data\' = \'%s\' not valid, must be inside: \'%s\'...' % (args.type_data, LIST_TYPE_DATA_AVAIL)
        catch_error_exception(message)

    if args.type_distribute not in LIST_TYPE_DISTRIBUTE_AVAIL:
        message = 'Input param \'type_distribute\' = \'%s\' not valid, must be inside: \'%s\'...' %(args.type_distribute, LIST_TYPE_DISTRIBUTE_AVAIL)
        catch_error_exception(message)

    if args.type_distribute == 'orderfile':
        if not args.infile_train_order:
            message = 'Input \'infile_train_order\' file for \'orderfile\' data distribution is needed...'
            catch_error_exception(message)
        else:
            args.infile_valid_order = args.infile_train_order.replace('train', 'valid')
            args.infile_test_order = args.infile_train_order.replace('train', 'test')

            if not is_exist_file(args.infile_train_order):
                message = 'Input \'infile_train_order\' file does not exist: \'%s\'...' %(args.infile_train_order)
                catch_error_exception(message)
            if not is_exist_file(args.infile_valid_order):
                message = 'Input \'infile_valid_order\' file does not exist: \'%s\'...' %(args.infile_valid_order)
                catch_error_exception(message)
            if not is_exist_file(args.infile_test_order):
                message = 'Input \'infile_test_order\' file does not exist: \'%s\'...' %(args.infile_test_order)
                catch_error_exception(message)

    if args.type_distribute == 'crossval' or \
       args.type_distribute == 'crossval_random':
        if not args.num_folds_crossval:
            message = 'Input \'num_folds_crossval\' for \'crossval\' data distribution is needed...'
            catch_error_exception(message)
        else:
            args.name_training_data_relpath   = set_dirname_suffix(args.name_training_data_relpath, 'CV%0.2i')
            args.name_validation_data_relpath = set_dirname_suffix(args.name_validation_data_relpath, 'CV%0.2i')
            args.name_testing_data_relpath    = set_dirname_suffix(args.name_testing_data_relpath, 'CV%0.2i')
            args.name_cvfolds_info_relpath    = 'CVfoldsInfo/'


    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)