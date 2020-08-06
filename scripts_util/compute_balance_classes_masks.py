
from common.functionutil import *
from dataloaders.imagefilereader import ImageFileReader
from imageoperators.imageoperator import *
import argparse


value_exclude = -1
value_foreground = 1
value_background = 0

def compute_balance_classes(in_mask):
    numvox_foreground_class = len(np.where(in_mask == value_foreground)[0])
    numvox_background_class = len(np.where(in_mask != value_foreground)[0])
    return (numvox_foreground_class, numvox_background_class)

def compute_balance_classes_with_exclusion(in_mask):
    numvox_exclude_class    = len(np.where(in_mask == value_exclude)[0])
    numvox_foreground_class = len(np.where(in_mask == value_foreground)[0])
    numvox_background_class = len(np.where(in_mask == value_background)[0])
    return (numvox_foreground_class, numvox_background_class)



def main(args):

    list_input_masks_files = list_files_dir(args.inputdir)

    list_ratio_back_foreground_class = []

    for i, in_mask_file in enumerate(list_input_masks_files):
        print("\nInput: \'%s\'..." % (basename(in_mask_file)))

        in_mask = ImageFileReader.get_image(in_mask_file)

        if (args.is_mask_region_interest):
            print("Compute ratio foreground / background masks with exclusion to Region of Interest...")
            (num_foreground_class, num_background_class) = compute_balance_classes_with_exclusion(in_mask)
        else:
            (num_foreground_class, num_background_class) = compute_balance_classes(in_mask)

        ratio_back_foreground_class = num_background_class / num_foreground_class

        list_ratio_back_foreground_class.append(ratio_back_foreground_class)

        print("Number of voxels of foreground masks: \'%s\', and background masks: \'%s\'..." %(num_foreground_class, num_background_class))
        print("Balance classes background / foreground masks: \'%s\'..." %(ratio_back_foreground_class))
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
        print("\'%s\' = %s" %(key, value))

    main(args)