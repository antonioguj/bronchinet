
import numpy as np
import argparse

from common.functionutil import basename, list_files_dir, str2bool
from dataloaders.imagefilereader import ImageFileReader

value_exclude = -1
value_foregrnd = 1
value_backgrnd = 0


def compute_balance_classes(in_mask):
    numvox_foregrnd_cls = len(np.where(in_mask == value_foregrnd)[0])
    numvox_backgrnd_cls = len(np.where(in_mask != value_foregrnd)[0])
    return (numvox_foregrnd_cls, numvox_backgrnd_cls)


def compute_balance_classes_with_exclusion(in_mask):
    # numvox_exclude_cls = len(np.where(in_mask == value_exclude)[0])
    numvox_foregrnd_cls = len(np.where(in_mask == value_foregrnd)[0])
    numvox_backgrnd_cls = len(np.where(in_mask == value_backgrnd)[0])
    return (numvox_foregrnd_cls, numvox_backgrnd_cls)


def main(args):

    list_input_masks_files = list_files_dir(args.inputdir)

    list_ratio_back_foreground_class = []

    for i, in_mask_file in enumerate(list_input_masks_files):
        print("\nInput: \'%s\'..." % (basename(in_mask_file)))

        in_mask = ImageFileReader.get_image(in_mask_file)

        if args.is_mask_region_interest:
            print("Compute ratio foreground / background masks with exclusion to Region of Interest...")
            (num_foregrnd_cls, num_backgrnd_cls) = compute_balance_classes_with_exclusion(in_mask)
        else:
            (num_foregrnd_cls, num_backgrnd_cls) = compute_balance_classes(in_mask)

        ratio_back_foreground_class = num_backgrnd_cls / num_foregrnd_cls

        list_ratio_back_foreground_class.append(ratio_back_foreground_class)

        print("Number voxels of foreground \'%s\' and background mask \'%s\'..." % (num_foregrnd_cls, num_backgrnd_cls))
        print("Balance classes background / foreground masks: \'%s\'..." % (ratio_back_foreground_class))
    # endfor

    average_ratio_back_foreground_class = sum(list_ratio_back_foreground_class) / len(list_ratio_back_foreground_class)

    print("\nAverage balance classes negative / positive: \'%s\'..." % (average_ratio_back_foreground_class))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str)
    parser.add_argument('--is_mask_region_interest', type=str2bool, default=True)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
