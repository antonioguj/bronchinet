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
    nameInputMasksRelPath  = 'Airways_Full'
    nameLungsMasksRelPath  = 'Lungs_Full'
    nameCentrelinesRelPath = 'RawAddMasks_Nosearch'
    nameOutputMasksRelPath = 'ProcAllMasks'

    nameMasksFiles       = '*surface0.nii.gz'
    nameLungsMasksFiles  = '*-lungs.nii.gz'
    nameCentrelinesFiles = '*_centrelines.dcm'

    tempSearchInputFiles = 'vol[0-9][0-9]_*'

    def rename_airways_files(in_name):
        return in_name.replace('surface0','lumen').replace('surface1','outerwall')

    nameOutAirwaysMasksFiles           = lambda in_name: rename_airways_files(filenamenoextension(in_name)) + '.nii.gz'
    nameOutLungsMasksFiles             = lambda in_name: filenamenoextension(in_name).replace('-lungs','_lungs') + '.nii.gz'
    nameOutTracheaMasksFiles           = lambda in_name: rename_airways_files(filenamenoextension(in_name)) + '_trachea.nii.gz'
    nameOutAirwaysPlusTracheaMasksFiles= lambda in_name: rename_airways_files(filenamenoextension(in_name)) + '_withtrachea.nii.gz'
    nameOutLungsPlusTracheaMasksFiles  = lambda in_name: rename_airways_files(filenamenoextension(in_name)) + '_lung-trachea.nii.gz'

    nameOutCentrelinesFiles            = lambda in_name: filenamenoextension(in_name).replace('_centrelines','_centrelines') + '.nii.gz'
    nameOutCentrelinesPlusTracheaFiles = lambda in_name: filenamenoextension(in_name).replace('_centrelines','_centreline-trachea') + '.nii.gz'
    # ---------- SETTINGS ----------



    workDirsManager = WorkDirsManager(args.basedir)
    BaseDataPath    = workDirsManager.getNameBaseDataPath()
    InputMasksPath  = workDirsManager.getNameExistPath(BaseDataPath, nameInputMasksRelPath)
    OutputMasksPath = workDirsManager.getNameNewPath  (BaseDataPath, nameOutputMasksRelPath)

    # Get the file list:
    listMasksFiles = findFilesDir(InputMasksPath, nameMasksFiles)
    nbMasksFiles   = len(listMasksFiles)

    # Run checkers
    if (nbMasksFiles == 0):
        message = "0 Images found in dir \'%s\'" %(InputMasksPath)
        CatchErrorException(message)


    if (args.masksToRegionInterest and isExistdir(joinpathnames(BaseDataPath, nameLungsMasksRelPath))):

        LungsMasksPath = workDirsManager.getNameExistPath(BaseDataPath, nameLungsMasksRelPath)

        listLungsMasksFiles = findFilesDir(LungsMasksPath, nameLungsMasksFiles)
        nbLungsMasksFiles   = len(listLungsMasksFiles)

        if nbLungsMasksFiles>0:
            isExistsLungsMasks = True
            print("Found Lungs Masks files in dir \'%s\'" % (LungsMasksPath))

            if (nbMasksFiles != nbLungsMasksFiles):
               message = "num Masks %s not equal to num Lungs Masks %s" %(nbMasksFiles, nbLungsMasksFiles)
               CatchErrorException(message)
        else:
            isExistsLungsMasks = False
    else:
        isExistsLungsMasks = False


    if (isExistdir(joinpathnames(BaseDataPath, nameCentrelinesRelPath))):

        CentrelinesPath = workDirsManager.getNameExistPath(BaseDataPath, nameCentrelinesRelPath)

        listCentrelinesFiles = findFilesDir(CentrelinesPath, nameCentrelinesFiles)
        nbCentrelinesFiles   = len(listCentrelinesFiles)

        if nbCentrelinesFiles > 0:
            isExistsCentrelines = True
            print("Found Centrelines files in dir \'%s\'" % (CentrelinesPath))

            if (nbMasksFiles != nbCentrelinesFiles):
                message = "num Masks %s not equal to num Centrelines %s" % (nbMasksFiles, nbCentrelinesFiles)
                CatchErrorException(message)
        else:
            isExistsCentrelines = False
    else:
        isExistsCentrelines = False


#    if (isExistsLungsMasks or isExistsCentrelines):
#        OutputMasksPath = workDirsManager.getNameNewPath(BaseDataPath, nameOutputAllMasksRelPath)
#    else:
#        OutputMasksPath = workDirsManager.getNameNewPath(BaseDataPath, nameOutputMasksRelPath)



    for i, masks_file in enumerate(listMasksFiles):

        print('\'%s\'...' %(masks_file))
        if '_surface0' in masks_file:
            print("Masks corresponding to Airways Lumen...")
        else:
            print("Masks corresponding to Airways Outer-wall...")

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


        if (isExistsLungsMasks):
            print("Preprocess Masks of Lungs and combinations with Airways Masks...")

            name_prefix_case = getExtractSubstringPattern(basename(masks_file),
                                                          tempSearchInputFiles)

            for iterfile, in zip(listLungsMasksFiles):
                if name_prefix_case in iterfile:
                    lungs_masks_file = iterfile

            print("assigned to: '%s'..." %(basename(lungs_masks_file)))

            lungs_masks_array = FileReader.getImageArray(lungs_masks_file)

            if (args.invertImageAxial):
                lungs_masks_array = FlippingImages.compute(lungs_masks_array, axis=0)

            # Turn to binary masks (0, 1)
            lungs_masks_array = OperationsBinaryMasks.process_masks(lungs_masks_array)

            # Extract voxels from ground-truth airways not contained in lungs
            trachea_masks_array = np.where(lungs_masks_array == 0, masks_array, 0)

            airwaysplustrachea_masks_array = masks_array

            # Exclude voxels from ground-truth airways not contained in lungs
            airways_masks_array = OperationsBinaryMasks.apply_mask_exclude_voxels_fillzero(masks_array, lungs_masks_array)

            # Join both lungs and trachea masks
            lungsplustrachea_masks_array = OperationsBinaryMasks.join_two_binmasks_one_image(lungs_masks_array, trachea_masks_array)

            masks_array = airways_masks_array


        if (isExistsCentrelines):
            print("Preprocess Centrelines of Airways...")

            centrelines_file = listCentrelinesFiles[i]

            print("assigned to: '%s'..." % (basename(centrelines_file)))

            centrelines_array = FileReader.getImageArray(centrelines_file)

            if (args.invertImageAxial):
                centrelines_array = FlippingImages.compute(centrelines_array, axis=0)

            # Turn to binary masks (0, 1)
            centrelines_array = OperationsBinaryMasks.process_masks(centrelines_array)

            if (isExistsLungsMasks):
                centrelinesplustrachea_array = centrelines_array

                # Exclude voxels from centrelines not contained in lungs
                centrelines_array = OperationsBinaryMasks.apply_mask_exclude_voxels_fillzero(centrelines_array, lungs_masks_array)



        print("Saving images in nifty '.nii' format of final dimensions: %s..." % (str(masks_array.shape)))

        out_masks_filename = joinpathnames(OutputMasksPath, basename(nameOutAirwaysMasksFiles(masks_file)))

        print (out_masks_filename)

        FileReader.writeImageArray(out_masks_filename, masks_array.astype(FORMATIMAGEDATA))

        if (isExistsLungsMasks):

            out_lungs_masks_filename        = joinpathnames(OutputMasksPath, nameOutLungsMasksFiles(lungs_masks_file))
            out_trachea_masks_filename      = joinpathnames(OutputMasksPath, nameOutTracheaMasksFiles(masks_file))
            out_airwaytrachea_masks_filename= joinpathnames(OutputMasksPath, nameOutAirwaysPlusTracheaMasksFiles(masks_file))
            out_lungstrachea_masks_filename = joinpathnames(OutputMasksPath, nameOutLungsPlusTracheaMasksFiles(masks_file))

            print (out_lungs_masks_filename)
            print (out_trachea_masks_filename)
            print (out_airwaytrachea_masks_filename)
            print (out_lungstrachea_masks_filename)

            FileReader.writeImageArray(out_lungs_masks_filename,         lungs_masks_array.astype(FORMATIMAGEDATA))
            FileReader.writeImageArray(out_trachea_masks_filename,       trachea_masks_array.astype(FORMATIMAGEDATA))
            FileReader.writeImageArray(out_airwaytrachea_masks_filename, airwaysplustrachea_masks_array.astype(FORMATIMAGEDATA))
            FileReader.writeImageArray(out_lungstrachea_masks_filename,  lungsplustrachea_masks_array.astype(FORMATIMAGEDATA))

        if (isExistsCentrelines):

            out_centrelines_filename = joinpathnames(OutputMasksPath, nameOutCentrelinesFiles(centrelines_file))

            FileReader.writeImageArray(out_centrelines_filename, centrelines_array.astype(FORMATIMAGEDATA))

            if (isExistsLungsMasks):

                out_centrelinestrachea_filename = joinpathnames(OutputMasksPath, nameOutCentrelinesPlusTracheaFiles(centrelines_file))

                FileReader.writeImageArray(out_centrelinestrachea_filename, centrelinesplustrachea_array.astype(FORMATIMAGEDATA))
    #endfor



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--multiClassCase', type=str2bool, default=MULTICLASSCASE)
    parser.add_argument('--numClassesMasks', type=int, default=NUMCLASSESMASKS)
    parser.add_argument('--invertImageAxial', type=str2bool, default=INVERTIMAGEAXIAL)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
