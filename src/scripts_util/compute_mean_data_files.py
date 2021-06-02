
import numpy as np
import argparse

from common.functionutil import is_exist_file
from common.exceptionmanager import catch_error_exception


def main(args):

    if args.fromfile:
        if not is_exist_file(args.list_input_files):
            message = "File \'%s\' not found..." % (args.list_input_files)
            catch_error_exception(message)

        with open(args.list_input_files, 'r') as fout:
            list_input_files = [infile.replace('\n', '') for infile in fout.readlines()]
        print("\'input_files\' = %s" % (list_input_files))
    else:
        list_input_files = [infile.replace('\n', '') for infile in args.input_files]
    num_input_files = len(list_input_files)

    print("Files to compute the mean from: \'%s\'..." % (num_input_files))
    for i, ifile in enumerate(list_input_files):
        print("%s: \'%s\'" % (i + 1, ifile))
    # endfor

    # ******************************

    if num_input_files == 1:
        print("Compute the mean of fields for cases from a single file...")

        raw_data_string = np.genfromtxt(list_input_files[0], dtype=str, delimiter=', ')
        raw_data_float = np.genfromtxt(list_input_files[0], dtype=float, delimiter=', ')

        header_file = list(raw_data_string[0, :])
        rows1elem_file = list(raw_data_string[:, 0])
        data = raw_data_float[1:, 1:]

        fields_names = [item.replace('/', '') for item in header_file[1:]]
        cases_names = rows1elem_file[1:]

        print("Compute mean data for fields: \'%s\'... and from cases: \'%s\'..." % (fields_names, cases_names))

        # Compute mean of data along the first dimension of array (cases)
        mean_data_cases = np.mean(data, axis=0)

        # Output mean results
        for i, ifield in enumerate(fields_names):
            print("Mean of \'%s\': %0.6f..." % (ifield, mean_data_cases[i]))
        # endfor

    else:
        print("Compute the mean of all data from \'%s\' files: %s..." % (num_input_files, list_input_files))

        header_1stfile = None
        rows1elem_1stfile = None
        list_data_files = None

        for (i, in_file) in enumerate(list_input_files):

            raw_data_this_string = np.genfromtxt(in_file, dtype=str, delimiter=', ')
            raw_data_this_float = np.genfromtxt(in_file, dtype=float, delimiter=', ')

            header_this = list(raw_data_this_string[0, :])
            rows1elem_this = list(raw_data_this_string[:, 0])
            data_this = raw_data_this_float[1:, 1:]

            if i == 0:
                header_1stfile = header_this
                rows1elem_1stfile = rows1elem_this
                # allocate vars to store data in files and compute the mean
                num_rows = len(rows1elem_this)
                num_cols = len(header_this)
                list_data_files = np.zeros((num_input_files, num_rows - 1, num_cols - 1))

            if header_this != header_1stfile:
                message = 'header in file: \'%s\' not equal to header found previously: \'%s\'' \
                          % (header_this, header_1stfile)
                catch_error_exception(message)
            if rows1elem_this != rows1elem_1stfile:
                message = '1st column in file: \'%s\' not equal to 1st column found previously: \'%s\'' \
                          % (rows1elem_this, rows1elem_1stfile)
                catch_error_exception(message)

            # store data corresponding to this file
            list_data_files[i, :, :] = data_this
        # endfor

        # Compute mean of data along the first dimension of array (input files)
        mean_data_fileslist = np.mean(list_data_files, axis=0)

        print("Save mean results in file: \'%s\'..." % (args.output_file))
        with open(args.output_file, 'w') as fout:

            strheader = ', '.join(header_1stfile) + '\n'
            fout.write(strheader)

            num_rows = len(rows1elem_1stfile)
            for i in range(num_rows - 1):
                data_thisrow = mean_data_fileslist[i]
                strdata = ', '.join([rows1elem_1stfile[i + 1]] + ['%0.6f' % (elem) for elem in data_thisrow]) + '\n'
                fout.write(strdata)
            # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', type=str, nargs='*')
    parser.add_argument('--output_file', type=str, default='results_mean.txt')
    parser.add_argument('--fromfile', type=bool, default=False)
    parser.add_argument('--list_input_files', type=str, default='list_input_files.txt')
    args = parser.parse_args()

    if args.fromfile and not args.list_input_files:
        message = 'need to input \'list_input_files\' with filenames to plot'
        catch_error_exception(message)

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
