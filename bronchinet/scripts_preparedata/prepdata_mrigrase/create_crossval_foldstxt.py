
import numpy as np
import argparse

from common.constant import DATADIR, NAME_REFERENCE_KEYS_PROCIMAGE_FILE
from common.functionutil import join_path_names, get_substring_filename, read_dictionary, str2bool

np.random.seed(2017)


def main(args):

    in_reference_keys_file = join_path_names(args.datadir, args.name_input_reference_keys_file)

    out_filename_cvfold_info_train = 'train%0.2i.txt'
    out_filename_cvfold_info_valid = 'valid%0.2i.txt'
    out_filename_cvfold_info_test = 'test%0.2i.txt'

    indict_reference_keys = read_dictionary(in_reference_keys_file)
    list_input_files = list(indict_reference_keys.values())

    # remove from the list of input files those that are from the same subject, and put them in a separate list
    list_in_unique_subject_files = []
    list_in_allfiles_same_subject = {}
    list_found_subjects = []
    for in_file in list_input_files:
        subject_name = get_substring_filename(in_file, 'Sujeto[0-9][0-9]')
        if subject_name not in list_found_subjects:
            list_found_subjects.append(subject_name)
            list_in_unique_subject_files.append(in_file)
            list_in_allfiles_same_subject[subject_name] = [in_file]
        else:
            list_in_allfiles_same_subject[subject_name].append(in_file)
    # endfor
    list_input_files = list_in_unique_subject_files

    # ******************************

    num_total_files = len(list_input_files)
    indexes_input_files = np.arange(num_total_files)

    if args.is_shuffle:
        np.random.shuffle(indexes_input_files)

    list_indexes_files_split_cvfolds = np.array_split(indexes_input_files, args.num_folds)

    num_testing_files_cvfolds = len(list_indexes_files_split_cvfolds[0])
    num_trainvalid_files_cvfolds = num_total_files - num_testing_files_cvfolds
    num_validation_files_cvfolds = np.int(args.prop_valid_in_training_folds * num_trainvalid_files_cvfolds)
    num_training_files_cvfolds = num_trainvalid_files_cvfolds - num_validation_files_cvfolds

    print("Num files (from unique subject) for Training (%s) / Validation (%s) / Testing (%s), in each cv-fold..."
          % (num_training_files_cvfolds, num_validation_files_cvfolds, num_testing_files_cvfolds))

    # to get ORDERED indexes for training + validation files in cv-folds
    indexes_input_files_repeated = np.concatenate((indexes_input_files, indexes_input_files))

    for i, indexes_files_split_cvfold in enumerate(list_indexes_files_split_cvfolds):

        pos_last_file_split_in_indexes = list(indexes_input_files).index(indexes_files_split_cvfold[-1])
        indexes_testing_files = indexes_files_split_cvfold
        indexes_trainvalid_files = \
            indexes_input_files_repeated[pos_last_file_split_in_indexes + 1:
                                         pos_last_file_split_in_indexes + 1 + num_trainvalid_files_cvfolds]
        indexes_training_files = indexes_trainvalid_files[:num_training_files_cvfolds]
        indexes_validation_files = indexes_trainvalid_files[num_training_files_cvfolds:]

        list_training_files = [list_input_files[ind] for ind in indexes_training_files]
        list_validation_files = [list_input_files[ind] for ind in indexes_validation_files]
        list_testing_files = [list_input_files[ind] for ind in indexes_testing_files]

        out_file_cvfold_info_train = out_filename_cvfold_info_train % (i + 1)
        out_file_cvfold_info_valid = out_filename_cvfold_info_valid % (i + 1)
        out_file_cvfold_info_test = out_filename_cvfold_info_test % (i + 1)

        print("For cv-fold %s, write distribution of files for Training / Validation / Testing in: %s, %s, %s..."
              % (i + 1, out_file_cvfold_info_train, out_file_cvfold_info_valid, out_file_cvfold_info_test))

        with open(out_file_cvfold_info_train, 'w') as fout:
            for in_unique_subject_file in list_training_files:
                subject_name = get_substring_filename(in_unique_subject_file, 'Sujeto[0-9][0-9]')
                list_in_same_subject_files = list_in_allfiles_same_subject[subject_name]
                for in_file in list_in_same_subject_files:
                    fout.write('%s\n' % (in_file))
            # endfor

        with open(out_file_cvfold_info_valid, 'w') as fout:
            for in_unique_subject_file in list_validation_files:
                subject_name = get_substring_filename(in_unique_subject_file, 'Sujeto[0-9][0-9]')
                list_in_same_subject_files = list_in_allfiles_same_subject[subject_name]
                for in_file in list_in_same_subject_files:
                    fout.write('%s\n' % (in_file))
            # endfor

        with open(out_file_cvfold_info_test, 'w') as fout:
            for in_unique_subject_file in list_testing_files:
                subject_name = get_substring_filename(in_unique_subject_file, 'Sujeto[0-9][0-9]')
                list_in_same_subject_files = list_in_allfiles_same_subject[subject_name]
                for in_file in list_in_same_subject_files:
                    fout.write('%s\n' % (in_file))
            # endfor
    # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    parser.add_argument('--num_folds', type=int, default=1)
    parser.add_argument('--is_shuffle', type=str2bool, default=True)
    parser.add_argument('--prop_valid_in_training_folds', type=float, default=0.2)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
