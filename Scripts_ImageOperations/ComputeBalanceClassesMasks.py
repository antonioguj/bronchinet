#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.Constants import *
from Common.WorkDirsManager import *
from DataLoaders.FileReaders import *
from OperationImages.OperationImages import *
import argparse

value_exclude = -1
value_foregrnd = 1
value_backgrnd = 0

def compute_balance_classes(mask_array):
    numvox_foregrnd_class = len(np.where(mask_array == value_foregrnd)[0])
    numvox_backgrnd_class = len(np.where(mask_array != value_foregrnd)[0])
    return (numvox_foregrnd_class, numvox_backgrnd_class)

def compute_balance_classes_with_exclusion(mask_array):
    numvox_exclude_class = len(np.where(mask_array == value_exclude)[0])
    numvox_foregrnd_class = len(np.where(mask_array == value_foregrnd)[0])
    numvox_backgrnd_class = len(np.where(mask_array == value_backgrnd)[0])
    return (numvox_foregrnd_class, numvox_backgrnd_class)



def main(args):
    # ---------- SETTINGS ----------
    nameInputRelPath = 'LabelsWorkData/'
    # ---------- SETTINGS ----------

    workDirsManager = WorkDirsManager(args.datadir)
    InputMasksPath  = workDirsManager.getNameExistPath(nameInputRelPath)

    listInputMasksFiles = findFilesDirAndCheck(InputMasksPath)


    list_ratio_back_foregrnd_class = []

    for i, in_mask_file in enumerate(listInputMasksFiles):
        print("\nInput: \'%s\'..." % (basename(in_mask_file)))

        in_mask_array  = FileReader.getImageArray(in_mask_file)

        if (args.masksToRegionInterest):
            (num_foregrnd_class, num_backgrnd_class) = compute_balance_classes_with_exclusion(in_mask_array)
        else:
            (num_foregrnd_class, num_backgrnd_class) = compute_balance_classes(in_mask_array)

        ratio_back_foregrnd_class = num_backgrnd_class / num_foregrnd_class

        list_ratio_back_foregrnd_class.append(ratio_back_foregrnd_class)

        print("Number of voxels of foreground masks: \'%s\', and background masks: \'%s\'..." %(num_foregrnd_class, num_backgrnd_class))
        print("Balance classes background / foreground masks: \'%s\'..." %(ratio_back_foregrnd_class))
    # endfor


    average_ratio_back_foregrnd_class = sum(list_ratio_back_foregrnd_class) / len(list_ratio_back_foregrnd_class)

    print("Average balance classes negative / positive: \'%s\'..." % (average_ratio_back_foregrnd_class))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)