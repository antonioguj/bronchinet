
import numpy as np
from scipy.stats import ttest_ind, ttest_rel
import argparse

from common.exceptionmanager import catch_error_exception

LIST_TYPE_PTESTS_AVAIL = ['independent', 'related']


def main(args):

    if args.type_ptest == 'independent':
        fun_ttest = ttest_ind
    elif args.type_ptest == 'related':
        fun_ttest = ttest_rel
    else:
        fun_ttest = None

    print("\nCompare the field \'%s\' between two data, using \'%s\' P-test..." % (args.in_fieldname, args.type_ptest))

    list_input_files = [args.inputfile_1, args.inputfile_2]
    list_input_data = []
    list_calc_stats_data = []

    for input_file in list_input_files:

        raw_data_this_string = np.genfromtxt(input_file, dtype=str, delimiter=', ')
        raw_data_this_float = np.genfromtxt(input_file, dtype=float, delimiter=', ')

        fields_names_this = [item.replace('/', '') for item in raw_data_this_string[0, 1:]]

        if args.in_fieldname not in fields_names_this:
            message = 'field \'%s\' not found in file \'%s\'' % (args.in_fieldname, input_file)
            catch_error_exception(message)
        else:
            index_infieldname = fields_names_this.index(args.in_fieldname) + 1
            data_this = raw_data_this_float[1:, index_infieldname]
            list_input_data.append(data_this)

            median_data_this = np.median(data_this)
            perc25_data_this = np.percentile(data_this, 25)
            perc75_data_this = np.percentile(data_this, 75)
            stats_data_this = (median_data_this, perc25_data_this, perc75_data_this)
            list_calc_stats_data.append(stats_data_this)
    # endfor

    (pstats_diff, pvalue_diff) = fun_ttest(list_input_data[0], list_input_data[1])

    print("data 1: %.3f (%.3f - %.3f)..." % (list_calc_stats_data[0][:]))
    print("data 2: %.3f (%.3f - %.3f)..." % (list_calc_stats_data[1][:]))
    print("Computed P-value (d1 - d2): %.6f..." % (pvalue_diff))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile_1', type=str, default='.')
    parser.add_argument('inputfile_2', type=str, default='.')
    parser.add_argument('in_fieldname', type=str, default='.')
    parser.add_argument('--type_ptest', type=str, default='related')
    args = parser.parse_args()

    if args.type_ptest not in LIST_TYPE_PTESTS_AVAIL:
        message = 'Input param \'type_ptest\' = \'%s\' not valid, must be: \'%s\'...' \
                  % (args.type_ptest, LIST_TYPE_PTESTS_AVAIL)
        catch_error_exception(message)

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
