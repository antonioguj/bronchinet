
import argparse

from common.functionutil import join_path_names, basename, list_files_dir, str2bool


def main(args):

    list_input_files = list_files_dir(args.inputdir)

    list_files_failed_rename_by_hand = []

    for i, in_file in enumerate(list_input_files):
        print("\nInput: \'%s\'..." % (in_file))

        in_file_basename = basename(in_file)
        in_file_extension = '.nii.gz'
        in_file_basename_noext = in_file_basename.replace(in_file_extension, '')

        in_file_list_splitname = in_file_basename_noext.split('_')

        if args.is_simulated_images:
            if len(in_file_list_splitname) != 7:
                print('WARNING. Expected a filename with 7 strings separated by \'_\'. '
                      'Found %s... Rename this file by hand' % (len(in_file_list_splitname)))
                list_files_failed_rename_by_hand.append(in_file)
                continue

            in_file_cube_or_grase = in_file_list_splitname[0]
            in_file_simulated_or_original = in_file_list_splitname[1]
            in_file_twonumbers_first = in_file_list_splitname[2:4]
            in_file_subject = in_file_list_splitname[4]
            in_file_numbers_second = in_file_list_splitname[5]
            in_file_left_or_right = in_file_list_splitname[6]

            in_file_twonumbers_first[0] = in_file_twonumbers_first[0].replace('.', '-')
            in_file_twonumbers_first[1] = in_file_twonumbers_first[1].replace('.', '-')

            if len(in_file_subject) == 4:
                in_file_subject = in_file_subject[0:3] + '0' + in_file_subject[-1]

            out_file_list_splitname = \
                [in_file_subject + in_file_left_or_right] + [in_file_cube_or_grase] + \
                [in_file_simulated_or_original] + in_file_twonumbers_first + [in_file_numbers_second]

        else:
            # normal images / labels
            if len(in_file_list_splitname) != 5:
                print('WARNING. Expected a filename with 5 strings separated by \'_\'. Found %s... '
                      'Rename this file by hand' % (len(in_file_list_splitname)))
                list_files_failed_rename_by_hand.append(in_file)
                continue

            in_file_cube_or_grase = in_file_list_splitname[0]
            in_file_simulated_or_original = in_file_list_splitname[1]
            in_file_subject = in_file_list_splitname[2]
            in_file_numbers_second = in_file_list_splitname[3]
            in_file_left_or_right = in_file_list_splitname[4]

            if len(in_file_subject) == 4:
                in_file_subject = in_file_subject[0:3] + '0' + in_file_subject[-1]

            out_file_list_splitname = [in_file_subject + in_file_left_or_right] + [in_file_cube_or_grase] + \
                                      [in_file_simulated_or_original] + [in_file_numbers_second]

        out_file_basename_noext = '_'.join(out_file_list_splitname)
        out_file_basename = out_file_basename_noext + in_file_extension
        out_file = join_path_names(args.inputdir, out_file_basename)

        print("Output: \'%s\'..." % (out_file))
        # movefile(in_file, out_file)
    # endfor

    print('\nFound %s files that failed and need to be renamed by hand:\n' % (len(list_files_failed_rename_by_hand)))
    print(list_files_failed_rename_by_hand)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str)
    parser.add_argument('--is_simulated_images', type=str2bool, default=False)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
