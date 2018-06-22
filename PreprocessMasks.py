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

    workDirsManager  = WorkDirsManager(args.basedir)
    BaseDataPath     = workDirsManager.getNameBaseDataPath()
    OriginMasksPath  = workDirsManager.getNameExistPath(BaseDataPath, 'RawMasks')
    OutputMasksPath  = workDirsManager.getNameNewPath(BaseDataPath, 'ProcMasks')

    # Get the file list:
    nameMasksFiles = '*.dcm'
    listMasksFiles = findFilesDir(OriginMasksPath,  nameMasksFiles)
    nbMasksFiles   = len(listMasksFiles)

    outFilesExtension = '.nii'

    if (args.confineMasksToLungs):

        OriginAddMasksPath = workDirsManager.getNameExistPath(BaseDataPath, 'RawAddMasks')
        OutputAddMasksPath = workDirsManager.getNameNewPath(BaseDataPath, 'ProcAddMasks')

        nameAddMasksFiles = '*.dcm'
        listAddMasksFiles = findFilesDir(OriginAddMasksPath, nameAddMasksFiles)
        nbAddMasksFiles   = len(listAddMasksFiles)


    # Run checkers
    if (nbMasksFiles == 0):
        message = "num CTs Images found in dir \'%s\'" %(OriginMasksPath)
        CatchErrorException(message)

    if (args.confineMasksToLungs):
        if (nbMasksFiles != nbAddMasksFiles):
            message = "num CTs Images %s not equal to num Masks %s" %(nbMasksFiles, nbAddMasksFiles)
            CatchErrorException(message)



    for i, masksFile in enumerate(listMasksFiles):

        print('\'%s\'...' %(masksFile))

        masks_array = FileReader.getImageArray(masksFile)


        if (args.multiClassCase):
            # Check the correct multilabels in "masks_array"
            if not checkCorrectNumClassesInMasks(masks_array, args.numClassesMasks):
                message = "In multiclass case, found wrong values in masks array: %s..." %(np.unique(masks_array))
                CatchErrorException(message)
        else:
            # Turn to binary masks (0, 1)
            masks_array = processBinaryMasks(masks_array)


        if (args.confineMasksToLungs):
            print("Confine masks to exclude the area outside the lungs...")

            exclude_masksFile   = listAddMasksFiles[i]
            exclude_masks_array = FileReader.getImageArray(exclude_masksFile)

            # Turn to binary masks (0, 1)
            lungs_masks_array   = processBinaryMasks(exclude_masks_array)
            traquea_masks_array = np.where(lungs_masks_array == 0, masks_array, 0)

            masks_array = ExclusionMasks.computeInverse(masks_array, exclude_masks_array)


        print("Saving images in nifty '.nii' format of final dimensions: %s..." % (str(masks_array.shape)))

        out_masksFilename = joinpathnames(OutputMasksPath, filenamenoextension(masksFile) + outFilesExtension)

        FileReader.writeImageArray(out_masksFilename, masks_array.astype(FORMATIMAGEDATA))

        if (args.confineMasksToLungs):
            out_lungs_masksFilename   = joinpathnames(OutputAddMasksPath, filenamenoextension(exclude_masksFile) + outFilesExtension)
            out_traquea_masksFilename = out_lungs_masksFilename.replace('lungs','traquea')

            FileReader.writeImageArray(out_lungs_masksFilename,   lungs_masks_array  .astype(FORMATIMAGEDATA))
            FileReader.writeImageArray(out_traquea_masksFilename, traquea_masks_array.astype(FORMATIMAGEDATA))
    #endfor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--multiClassCase', type=str2bool, default=MULTICLASSCASE)
    parser.add_argument('--numClassesMasks', type=int, default=NUMCLASSESMASKS)
    parser.add_argument('--confineMasksToLungs', default=CONFINEMASKSTOLUNGS)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
