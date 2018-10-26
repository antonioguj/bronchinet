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
from Preprocessing.BoundingBoxMasks import *
from Preprocessing.OperationsImages import *
import argparse



def main(args):

    # ---------- SETTINGS ----------
    nameRawInputImagesRelPath      = 'RawImages_Full'
    nameRawInputMasksRelPath       = 'RawAirways'
    nameRawInputLungsMasksRelPath  = 'RawLungs'
    nameRawOutputMasksRelPath      = 'Airways_Full'
    nameRawOutputLungsMasksRelPath = 'Lungs_Full'

    # Get the file list:
    nameInputImagesFiles     = '*.dcm'
    nameInputMasksFiles      = '*surface0.dcm'
    nameInputLungsMasksFiles = '*-lungs.dcm'

    nameBoundingBoxesMasks = 'found_boundBoxes_Original.npy'

    nameOutputMasksFiles      = lambda in_name: filenamenoextension(in_name) + '.nii.gz'
    nameOutputLungsMasksFiles = lambda in_name: filenamenoextension(in_name) + '.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager    = WorkDirsManager(args.basedir)
    BaseDataPath       = workDirsManager.getNameBaseDataPath()
    RawInputImagesPath = workDirsManager.getNameExistPath(BaseDataPath, nameRawInputImagesRelPath)
    RawInputMasksPath  = workDirsManager.getNameExistPath(BaseDataPath, nameRawInputMasksRelPath)
    RawOutputMasksPath = workDirsManager.getNameNewPath  (BaseDataPath, nameRawOutputMasksRelPath)

    listInputImagesFiles = findFilesDir(RawInputImagesPath, nameInputImagesFiles)
    listInputMasksFiles  = findFilesDir(RawInputMasksPath,  nameInputMasksFiles)

    nbInputImagesFiles = len(listInputImagesFiles)
    nbInputMasksFiles  = len(listInputMasksFiles)

    # Run checkers
    if (nameInputMasksFiles == 0):
        message = "0 Cropped Masks found in dir \'%s\'" %(RawInputMasksPath)
        CatchErrorException(message)
    if (nbInputMasksFiles != nbInputImagesFiles):
        message = "num Cropped Masks %i not equal to num Images %i" %(nbInputMasksFiles, nbInputImagesFiles)
        CatchErrorException(message)


    if isExistdir(joinpathnames(BaseDataPath, nameRawInputLungsMasksRelPath)):
        isExistsLungsMasks = True

        RawInputLungsMasksPath  = workDirsManager.getNameExistPath(BaseDataPath, nameRawInputLungsMasksRelPath)
        RawOutputLungsMasksPath = workDirsManager.getNameNewPath  (BaseDataPath, nameRawOutputLungsMasksRelPath)

        listInputLungsMasksFiles = findFilesDir(RawInputLungsMasksPath, nameInputLungsMasksFiles)
        nbInputLungsMasksFiles   = len(listInputLungsMasksFiles)

        if (nbInputLungsMasksFiles != nbInputImagesFiles):
            message = "num Cropped Lungs Masks %i not equal to num Images %i" % (nbInputLungsMasksFiles, nbInputImagesFiles)
            CatchErrorException(message)
    else:
        isExistsLungsMasks = False

    namefile_dict = joinpathnames(BaseDataPath, nameBoundingBoxesMasks)
    dict_boundingBoxes = readDictionary(namefile_dict)



    for i, (masks_cropped_file, images_full_file) in enumerate(zip(listInputMasksFiles, listInputImagesFiles)):

        print('\'%s\'...' %(masks_cropped_file))

        masks_cropped_array = FileReader.getImageArray(masks_cropped_file)

        masks_cropped_array = FlippingImages.compute(masks_cropped_array, axis=0)

        print("dimensions of Cropped Masks: %s..." %(str(masks_cropped_array.shape)))

        if (isExistsLungsMasks):
            lungs_masks_cropped_file = listInputLungsMasksFiles[i]

            lungs_masks_cropped_array = FileReader.getImageArray(lungs_masks_cropped_file)

            lungs_masks_cropped_array = FlippingImages.compute(lungs_masks_cropped_array, axis=0)


        # Retrieve shape of full images
        images_full_shape = FileReader.getImageSize(images_full_file)

        masks_full_array = np.zeros(images_full_shape, dtype=FORMATMASKDATA)

        if (isExistsLungsMasks):
            lungs_masks_full_array = np.zeros(images_full_shape, dtype=FORMATMASKDATA)


        # Retrieve bounding-box of original cropped images
        bounding_box = dict_boundingBoxes[filenamenoextension(images_full_file)]

        masks_full_array[bounding_box[0][0]:bounding_box[0][1],
                         bounding_box[1][0]:bounding_box[1][1],
                         bounding_box[2][0]:bounding_box[2][1]] = masks_cropped_array

        if (isExistsLungsMasks):
            lungs_masks_full_array[bounding_box[0][0]:bounding_box[0][1],
                                   bounding_box[1][0]:bounding_box[1][1],
                                   bounding_box[2][0]:bounding_box[2][1]] = lungs_masks_cropped_array


        print("Saving Full Masks of final dimensions: %s..." % (str(masks_full_array.shape)))

        out_masks_full_filename = joinpathnames(RawOutputMasksPath, nameOutputMasksFiles(masks_cropped_file))

        FileReader.writeImageArray(out_masks_full_filename, masks_full_array)

        if (isExistsLungsMasks):
            out_lungs_masks_full_filename = joinpathnames(RawOutputLungsMasksPath, nameOutputLungsMasksFiles(lungs_masks_cropped_file))

            FileReader.writeImageArray(out_lungs_masks_full_filename, lungs_masks_full_array)
    #endfor



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)