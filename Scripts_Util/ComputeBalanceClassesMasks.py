#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from DataLoaders.FileReaders import *
from OperationImages.OperationImages import *
import argparse


value_exclude = -1
value_foreground = 1
value_background = 0

def compute_balance_classes(in_mask_array):
    numvox_foreground_class = len(np.where(in_mask_array == value_foreground)[0])
    numvox_background_class = len(np.where(in_mask_array != value_foreground)[0])
    return (numvox_foreground_class, numvox_background_class)

def compute_balance_classes_with_exclusion(in_mask_array):
    numvox_exclude_class    = len(np.where(in_mask_array == value_exclude)[0])
    numvox_foreground_class = len(np.where(in_mask_array == value_foreground)[0])
    numvox_background_class = len(np.where(in_mask_array == value_background)[0])
    return (numvox_foreground_class, numvox_background_class)



def main(args):

    listInputMasksFiles = findFilesDirAndCheck(args.inputdir)

    list_ratio_back_foreground_class = []

    for i, in_mask_file in enumerate(listInputMasksFiles):
        print("\nInput: \'%s\'..." % (basename(in_mask_file)))

        in_mask_array = FileReader.getImageArray(in_mask_file)

        if (args.masksToRegionInterest):
            print("Compute ratio foreground / background masks with exclusion to Region of Interest...")
            (num_foreground_class, num_background_class) = compute_balance_classes_with_exclusion(in_mask_array)
        else:
            (num_foreground_class, num_background_class) = compute_balance_classes(in_mask_array)

        ratio_back_foreground_class = num_background_class / num_foreground_class

        list_ratio_back_foreground_class.append(ratio_back_foreground_class)

        print("Number of voxels of foreground masks: \'%s\', and background masks: \'%s\'..." %(num_foreground_class, num_background_class))
        print("Balance classes background / foreground masks: \'%s\'..." %(ratio_back_foreground_class))
    # endfor


    average_ratio_back_foreground_class = sum(list_ratio_back_foreground_class) / len(list_ratio_back_foreground_class)

    print("Average balance classes negative / positive: \'%s\'..." % (average_ratio_back_foreground_class))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=True)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)