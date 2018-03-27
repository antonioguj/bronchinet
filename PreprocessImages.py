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
from CommonUtil.FileReaders import *
from CommonUtil.ErrorMessages import *
from CommonUtil.FunctionsUtil import *
from CommonUtil.WorkDirsManager import WorkDirsManager
from Preprocessing.ConfineMasksToLungs import *
from Preprocessing.OperationsImages import *
from Preprocessing.SlidingWindowImages import *
import numpy as np



#MAIN
workDirsManager  = WorkDirsManager(BASEDIR)
BaseDataPath     = workDirsManager.getNameDataPath(TYPEDATA)
ImagesPath       = workDirsManager.getNameNewPath(BaseDataPath, 'RawCTs')
MasksPath        = workDirsManager.getNameNewPath(BaseDataPath, 'RawMasks')
LungsMasksPath   = workDirsManager.getNameNewPath(BaseDataPath, 'RawLungs')
ProcVolsDataPath = workDirsManager.getNameNewPath(BaseDataPath, 'ProcVolsData')


# Get the file list:
listImagesFiles     = findFilesDir(ImagesPath    + '/*.dcm')
listMasksFiles      = findFilesDir(MasksPath     + '/*.dcm')
listLungsMasksFiles = findFilesDir(LungsMasksPath+ '/*.dcm')

nbImagesFiles     = len(listImagesFiles)
nbMasksFiles      = len(listMasksFiles)
nbLungsMasksFiles = len(listLungsMasksFiles)

# Run checkers
if (nbImagesFiles == 0):
    message = "num CTs Images found in dir \'%s\'" %(ImagesPath)
    CatchErrorException(message)
if (nbImagesFiles != nbMasksFiles or
    nbImagesFiles != nbLungsMasksFiles):
    message = "num CTs Images %i not equal to num Masks %i" %(nbImagesFiles, nbMasksFiles)
    CatchErrorException(message)


if (CROPPINGIMAGES):
    namefile_dict = joinpathnames(BaseDataPath, "boundBoxesMasks.npy")
    dict_masks_boundingBoxes = readDictionary(namefile_dict)


for i, (imagesFile, masksFile, lungs_masksFile) in enumerate(zip(listImagesFiles, listMasksFiles, listLungsMasksFiles)):

    print('\'%s\'...' %(imagesFile))

    images_array      = FileReader.getImageArray(imagesFile)
    masks_array       = FileReader.getImageArray(masksFile)
    lungs_masks_array = FileReader.getImageArray(lungs_masksFile)

    # Turn to binary masks (0, 1)
    masks_array = processBinaryMasks(masks_array)

    # Revert Images to start from Traquea
    images_array     = revertStackImages(images_array)
    masks_array      = revertStackImages(masks_array )
    lungs_masks_array= revertStackImages(lungs_masks_array)


    if (CONFINEMASKSTOLUNGS):

        masks_array = ConfineMasksToLungs.compute(masks_array, lungs_masks_array)

    if (CROPPINGIMAGES):

        crop_boundingBox = dict_masks_boundingBoxes[basename(imagesFile)]

        images_array = CropImages.compute3D(images_array, crop_boundingBox)
        masks_array  = CropImages.compute3D(masks_array,  crop_boundingBox)

    if (SAVEIMAGESINBATCHES):

        if (SLIDINGWINDOWIMAGES):

            images_array = SlidingWindowImages.compute(images_array, IMAGES_DIMS_Z_X_Y, PROP_OVERLAP_Z_X_Y)
            masks_array  = SlidingWindowImages.compute(masks_array,  IMAGES_DIMS_Z_X_Y, PROP_OVERLAP_Z_X_Y)

        else:
            images_array = StackBatchesImages.compute_stack_vols(images_array, IMAGES_DEPTHZ)
            masks_array  = StackBatchesImages.compute_stack_vols(masks_array,  IMAGES_DEPTHZ)


    stringinfo = "_".join(str(i) for i in list(images_array.shape))

    out_imagesFile = joinpathnames(ProcVolsDataPath, 'volsImages-%0.2i_dim'%(i)+stringinfo+'.npy')
    out_masksFile  = joinpathnames(ProcVolsDataPath, 'volsMasks-%0.2i_dim'%(i)+stringinfo+'.npy')

    np.save(out_imagesFile, images_array)
    np.save(out_masksFile,  masks_array )
#endfor