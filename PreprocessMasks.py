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
from Preprocessing.OperationsImages import *
import argparse



def main(args):

    # ---------- SETTINGS ----------
    nameOriginMasksRelPath    = 'ProcOrigMasks'
    nameOutputMasksRelPath    = 'ProcMasks'
    nameInputAddMasksRelPath  = 'ProcOrigAddMasks'
    nameOutputAddMasksRelPath = 'ProcAddMasks'

    nameMasksFiles = '*.nii.gz'
    nameExcludeMasksFiles = '*.nii.gz'

    nameOutMasksFiles           = lambda in_name: filenamenoextension(in_name) + '.nii.gz'
    nameOutAirwayMasksPlusTraqueaFiles= lambda in_name: filenamenoextension(in_name) + '_full.nii.gz'
    nameOutLungsMasksFiles      = lambda in_name: filenamenoextension(in_name) + '.nii.gz'
    nameOutTraqueaMasksFiles    = lambda in_name: filenamenoextension(in_name).replace('lungs','traquea')
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(args.basedir)
    BaseDataPath    = workDirsManager.getNameBaseDataPath()
    OriginMasksPath = workDirsManager.getNameExistPath(BaseDataPath, nameOriginMasksRelPath)
    OutputMasksPath = workDirsManager.getNameNewPath(BaseDataPath, nameOutputMasksRelPath)

    # Get the file list:
    listMasksFiles = findFilesDir(OriginMasksPath,  nameMasksFiles)
    nbMasksFiles   = len(listMasksFiles)

    # Run checkers
    if (nbMasksFiles == 0):
        message = "num CTs Images found in dir \'%s\'" %(OriginMasksPath)
        CatchErrorException(message)

    if (args.confineMasksToLungs):

        InputAddMasksPath  = workDirsManager.getNameExistPath(BaseDataPath, nameInputAddMasksRelPath)
        OutputAddMasksPath = workDirsManager.getNameNewPath(BaseDataPath, nameOutputAddMasksRelPath)

        listExcludeMasksFiles = findFilesDir(InputAddMasksPath, nameExcludeMasksFiles)
        nbExcludeMasksFiles   = len(listExcludeMasksFiles)

        if (nbMasksFiles != nbExcludeMasksFiles):
            message = "num CTs Images %s not equal to num Masks %s" %(nbMasksFiles, nbExcludeMasksFiles)
            CatchErrorException(message)



    for i, masks_file in enumerate(listMasksFiles):

        print('\'%s\'...' %(masks_file))

        masks_array = FileReader.getImageArray(masks_file)

        if (args.invertImageAxial):
            masks_array = FlippingImages.compute(masks_array,  axis=0)


        if (args.multiClassCase):
            operationsMasks = OperationsMultiClassMasks(args.numClassesMasks)

            # Check the correct labels in "masks_array"
            if not operationsMasks.check_masks(masks_array):
                message = "found wrong labels in masks array: %s..." %(np.unique(masks_array))
                CatchErrorException(message)

            message = "MULTICLASS CASE STILL IN IMPLEMENTATION...EXIT"
            CatchErrorException(message)
        else:
            # Turn to binary masks (0, 1)
            masks_array = OperationsBinaryMasks.process_masks(masks_array)


        if (args.confineMasksToLungs):
            print("Confine masks to exclude the area outside the lungs...")

            exclude_masks_file  = listExcludeMasksFiles[i]

            print("assigned to: '%s'..." %(basename(exclude_masks_file)))

            exclude_masks_array = FileReader.getImageArray(exclude_masks_file)

            if (args.invertImageAxial):
                exclude_masks_array = FlippingImages.compute(exclude_masks_array, axis=0)

            # Turn to binary masks (0, 1)
            lungs_masks_array   = OperationsBinaryMasks.process_masks(exclude_masks_array)
            traquea_masks_array = np.where(lungs_masks_array == 0, masks_array, 0)

            masks_plustraquea_array = masks_array

            masks_array = OperationsBinaryMasks.apply_mask_exclude_voxels_fillzero(masks_array, exclude_masks_array)


        print("Saving images in nifty '.nii' format of final dimensions: %s..." % (str(masks_array.shape)))

        out_masks_filename = joinpathnames(OutputMasksPath, nameOutMasksFiles(masks_file))

        FileReader.writeImageArray(out_masks_filename, masks_array.astype(FORMATIMAGEDATA))

        if (args.confineMasksToLungs):

            out_masks_plustraquea_filename = joinpathnames(OutputAddMasksPath, nameOutAirwayMasksPlusTraqueaFiles(masks_file))
            out_lungs_masks_filename       = joinpathnames(OutputAddMasksPath, nameOutLungsMasksFiles(exclude_masks_file))
            out_traquea_masks_filename     = joinpathnames(OutputAddMasksPath, nameOutTraqueaMasksFiles(basename(exclude_masks_file)))

            FileReader.writeImageArray(out_masks_plustraquea_filename, masks_plustraquea_array.astype(FORMATIMAGEDATA))
            FileReader.writeImageArray(out_lungs_masks_filename,       lungs_masks_array      .astype(FORMATIMAGEDATA))
            FileReader.writeImageArray(out_traquea_masks_filename,     traquea_masks_array    .astype(FORMATIMAGEDATA))
    #endfor



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--multiClassCase', type=str2bool, default=MULTICLASSCASE)
    parser.add_argument('--numClassesMasks', type=int, default=NUMCLASSESMASKS)
    parser.add_argument('--invertImageAxial', type=str2bool, default=INVERTIMAGEAXIAL)
    parser.add_argument('--confineMasksToLungs', type=str2bool, default=CONFINEMASKSTOLUNGS)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
