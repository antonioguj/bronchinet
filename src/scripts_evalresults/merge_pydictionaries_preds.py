
from collections import OrderedDict
import argparse

from common.functionutil import read_dictionary, save_dictionary, save_dictionary_csv


def main(args):

    output_dict_preds_referkeys_all = OrderedDict()

    for input_file in args.list_input_files:
        input_dict_preds_referkeys_this = read_dictionary(input_file)
        output_dict_preds_referkeys_all.update(input_dict_preds_referkeys_this)
    # endfor

    save_dictionary(args.output_file, output_dict_preds_referkeys_all)
    save_dictionary_csv(args.output_file.replace('.npy', '.csv'), output_dict_preds_referkeys_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('list_input_files', nargs='+', type=str)
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
