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
    nameRawMasksCroppedRelPath    = 'RawMasks_Cropped'
    nameRawImagesFullRelPath      = 'RawImages_Full'
    nameRawImagesCroppedRelPath   = 'RawImages_Cropped'
    nameRawOutMasksFullRelPath    = 'ProcMasks_Full'
    nameRawAddMasksCroppedRelPath = 'RawAddMasks_Cropped'
    nameRawOutAddMasksFullRelPath = 'ProcAllMasks_Full'

    # Get the file list:
    nameMasksCroppedFiles   = '*.dcm'
    nameImagesFullFiles     = '*.dcm'
    nameImagesCroppedFiles  = '*.dcm'
    nameAddMasksCroppedFiles= '*.dcm'

    nameBoundingBoxesMasks = 'found_boundBoxes_Original.npy'

    nameOutMasksFiles = lambda in_name: filenamenoextension(in_name) + '.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager     = WorkDirsManager(args.basedir)
    BaseDataPath        = workDirsManager.getNameBaseDataPath()
    RawMasksCroppedPath = workDirsManager.getNameExistPath(BaseDataPath, nameRawMasksCroppedRelPath)
    RawImagesFullPath   = workDirsManager.getNameExistPath(BaseDataPath, nameRawImagesFullRelPath  )
    RawImagesCroppedPath= workDirsManager.getNameExistPath(BaseDataPath, nameRawImagesCroppedRelPath)
    RawOutMasksFullPath = workDirsManager.getNameNewPath  (BaseDataPath, nameRawOutMasksFullRelPath)

    listMasksCroppedFiles = findFilesDir(RawMasksCroppedPath, nameMasksCroppedFiles)
    listImagesFullFiles   = findFilesDir(RawImagesFullPath,   nameImagesFullFiles)
    listImagesCroppedFiles= findFilesDir(RawImagesCroppedPath,nameImagesCroppedFiles)

    nbMasksCroppedFiles = len(listMasksCroppedFiles)
    nbImagesFullFiles   = len(listImagesFullFiles)

    # Run checkers
    if (nameMasksCroppedFiles == 0):
        message = "0 Cropped Images found in dir \'%s\'" %(RawMasksCroppedPath)
        CatchErrorException(message)
    if (nbMasksCroppedFiles != nbImagesFullFiles):
        message = "num Cropped Masks %i not equal to num Full Images %i" %(nbMasksCroppedFiles, nbImagesFullFiles)
        CatchErrorException(message)


    if isExistdir(joinpathnames(BaseDataPath, nameRawAddMasksCroppedRelPath)):
        isExistsAddMasks = True

        RawAddMasksCroppedPath = workDirsManager.getNameExistPath(BaseDataPath, nameRawAddMasksCroppedRelPath)
        RawOutAddMasksFullPath = workDirsManager.getNameNewPath  (BaseDataPath, nameRawOutAddMasksFullRelPath)

        listAddMasksCroppedFiles = findFilesDir(RawAddMasksCroppedPath, nameAddMasksCroppedFiles)
        nbAddMasksCroppedFiles   = len(listAddMasksCroppedFiles)

        if (nbAddMasksCroppedFiles != nbImagesFullFiles):
            message = "num Cropped Masks %i not equal to num Full Images %i" % (nbAddMasksCroppedFiles, nbImagesFullFiles)
            CatchErrorException(message)
    else:
        isExistsAddMasks = False

    namefile_dict = joinpathnames(BaseDataPath, nameBoundingBoxesMasks)
    dict_boundingBoxes = readDictionary(namefile_dict)



    for i, (masks_cropped_file, images_full_file) in enumerate(zip(listMasksCroppedFiles, listImagesFullFiles)):

        print('\'%s\'...' %(masks_cropped_file))

        masks_cropped_array = FileReader.getImageArray(masks_cropped_file)

        masks_cropped_array = FlippingImages.compute(masks_cropped_array, axis=0)

        print("dimensions of Cropped Masks: %s..." %(str(masks_cropped_array.shape)))

        images_full_shape = FileReader.getImageSize(images_full_file)

        masks_full_array = np.zeros(images_full_shape, dtype=FORMATMASKDATA)

        if (isExistsAddMasks):
            add_masks_cropped_file = listAddMasksCroppedFiles[i]

            add_masks_cropped_array = FileReader.getImageArray(add_masks_cropped_file)

            add_masks_cropped_array = FlippingImages.compute(add_masks_cropped_array, axis=0)

            add_masks_full_array = np.zeros(images_full_shape, dtype=FORMATMASKDATA)


        images_cropped_file = listImagesCroppedFiles[i]

        bounding_box = dict_boundingBoxes[filenamenoextension(images_cropped_file)]

        masks_full_array[bounding_box[0][0]:bounding_box[0][1],
                         bounding_box[1][0]:bounding_box[1][1],
                         bounding_box[2][0]:bounding_box[2][1]] = masks_cropped_array

        if (isExistsAddMasks):
            add_masks_full_array[bounding_box[0][0]:bounding_box[0][1],
                                 bounding_box[1][0]:bounding_box[1][1],
                                 bounding_box[2][0]:bounding_box[2][1]] = add_masks_cropped_array


        print("Saving Full Masks of final dimensions: %s..." % (str(masks_full_array.shape)))

        out_masks_filename = joinpathnames(RawOutMasksFullPath, nameOutMasksFiles(masks_cropped_file))

        FileReader.writeImageArray(out_masks_filename, masks_full_array)

        if (isExistsAddMasks):
            out_add_masks_filename = joinpathnames(RawOutAddMasksFullPath, nameOutMasksFiles(add_masks_cropped_file))

            FileReader.writeImageArray(out_add_masks_filename, add_masks_full_array)
    #endfor



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)