#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.Constants import *
from CommonUtil.ErrorMessages import *
from CommonUtil.FileReaders import *
from CommonUtil.FunctionsUtil import *
from CommonUtil.WorkDirsManager import *
from Preprocessing.BalanceClassesMasks import *
from Preprocessing.OperationsImages import *
import argparse



def main(args):

    # ---------- SETTINGS ----------
    nameInputMasksDataRelPath = 'ProcMasksExperData'
    #nameInputMasksDataRelPath = 'ProcMasks'
    #nameLungsMasksDataRelPath = 'ProcAllMasks'

    # Get the file list:
    nameMasksFiles = 'masks*' + getFileExtension(FORMATINOUTDATA)
    #nameMasksFiles = '*.nii.gz'
    #nameLungsMasksFiles = '*lungs.nii.gz'
    # ---------- SETTINGS ----------

    workDirsManager   = WorkDirsManager(args.basedir)
    BaseDataPath      = workDirsManager.getNameBaseDataPath()
    InputMasksDataPath= workDirsManager.getNameExistPath(workDirsManager.getNameDataPath(args.typedata))
    #InputMasksDataPath= workDirsManager.getNameExistPath(BaseDataPath, nameInputMasksDataRelPath)

    listMasksFiles = findFilesDir(InputMasksDataPath, nameMasksFiles)

    nbMasksFiles  = len(listMasksFiles)

    # if (args.masksRegionInterest):
    #
    #     LungsMasksDataPath = workDirsManager.getNameExistPath(BaseDataPath, nameLungsMasksDataRelPath)
    #
    #     listLungsMasksFiles = findFilesDir(LungsMasksDataPath, nameLungsMasksFiles)
    #     nbLungsMasksFiles   = len(listLungsMasksFiles)
    #
    #     if (nbMasksFiles != nbLungsMasksFiles):
    #         message = "num Masks %i not equal to num Lungs Masks %i" %(nbMasksFiles, nbLungsMasksFiles)
    #         CatchErrorException(message)


    list_ratio_back_foregrnd_class = []


    for i, masks_file in enumerate(listMasksFiles):

        print('\'%s\'...' %(masks_file))

        masks_array  = FileReader.getImageArray(masks_file)

        # if (args.masksToRegionInterest):
        #     print("Mask ground-truth to Region of Interest: exclude voxels outside ROI: lungs...")
        #
        #     lungs_masks_file  = listLungsMasksFiles[i]
        #     lungs_masks_array = FileReader.getImageArray(lungs_masks_file)
        #
        #     masks_array = OperationsBinaryMasks.apply_mask_exclude_voxels(masks_array, lungs_masks_array)


        if (args.masksToRegionInterest):

            (num_foregrnd_class, num_backgrnd_class) = BalanceClassesMasks.compute_with_exclusion(masks_array)
        else:
            (num_foregrnd_class, num_backgrnd_class) = BalanceClassesMasks.compute(masks_array)

        ratio_back_foregrnd_class = num_backgrnd_class / num_foregrnd_class

        list_ratio_back_foregrnd_class.append(ratio_back_foregrnd_class)


        print("Number of voxels of foreground masks: %s, and background masks: %s..." %(num_foregrnd_class, num_backgrnd_class))
        print("Balance classes background / foreground masks: %s..." %(ratio_back_foregrnd_class))
    # endfor


    average_ratio_back_foregrnd_class = sum(list_ratio_back_foregrnd_class) / len(list_ratio_back_foregrnd_class)

    print("Average balance classes negative / positive: %s..." % (average_ratio_back_foregrnd_class))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--typedata', default='training')
    parser.add_argument('--multiClassCase', type=str2bool, default=MULTICLASSCASE)
    parser.add_argument('--numClassesMasks', type=int, default=NUMCLASSESMASKS)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)