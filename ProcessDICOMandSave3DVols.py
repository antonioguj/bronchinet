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
from CommonUtil.WorkDirsManager import WorkDirsManager
from glob import glob
import numpy as np
import os

TYPEDATA = 'testing'

FORMATINPUTIMAGES = 'dicom'
CROPPINGIMAGES = True
AUGMENTDATATWOSIDES = True
CHECKMASKSOUTBOUNDBOX = False


def checkMaskOutBoundBox(image_array, croppatch):

    image_array[:,croppatch[0][0]:croppatch[0][1],
                  croppatch[1][0]:croppatch[1][1]] = 0
    return (np.count_nonzero(image_array) != 0)


#MAIN
workDirsManager = WorkDirsManager(BASEDIR)
BaseDataPath    = workDirsManager.getNameDataPath(TYPEDATA)
RawImagesPath   = workDirsManager.getNameRawImagesDataPath(TYPEDATA)
RawMasksPath    = workDirsManager.getNameRawMasksDataPath(TYPEDATA)
ProcVolsDataPath= workDirsManager.getNameNewPath(BaseDataPath, 'ProcVolsData')


if (CROPPINGIMAGES):
    sizeBoundBox = (CROPPATCH[0][1] - CROPPATCH[0][0],
                    CROPPATCH[1][1] - CROPPATCH[1][0])
    print('Cropping CT volumes to Bounding-Box of (%s,%s)' %(sizeBoundBox))

    if (sizeBoundBox[0] != IMAGES_HEIGHT or
        sizeBoundBox[1] != IMAGES_WIDTH):
        message = 'size of Input Images not equal to Bounding-Box size (%s,%s)' %(sizeBoundBox)
        CatchErrorException(message)

    if (AUGMENTDATATWOSIDES):
        CROPPATCHMIRROR = getCROPPATCHMIRROR(CROPPATCH)

    if (CHECKMASKSOUTBOUNDBOX):
        if (not AUGMENTDATATWOSIDES):
            CROPPATCHMIRROR = getCROPPATCHMIRROR(CROPPATCH)
        CROPPATHTWOSIDES = ((CROPPATCH[0][0], CROPPATCH[0][1]),
                            (CROPPATCH[1][0], CROPPATCHMIRROR[1][1]))


# Get the file list:
listImagesFiles = sorted(glob(RawImagesPath + '/*.dcm'))
listMasksFiles  = sorted(glob(RawMasksPath  + '/*.dcm'))

nbImagesFiles = len(listImagesFiles)
nbMasksFiles  = len(listMasksFiles)

if (nbImagesFiles == 0):
    message = "no Image files found in dir \'%s\'" %(RawImagesPath)
    CatchErrorException(message)
if (nbImagesFiles != nbMasksFiles):
    message = "nb Images files %i not equal to nb Masks files %i" %(nbImagesFiles,nbMasksFiles)
    CatchErrorException(message)


nbFiles = len(listImagesFiles)

print('-' * 30)
print('Loading CT volumes (%s in total) and Saving...' % (nbFiles))
print('-' * 30)

for i, (in_imagesFile, in_masksFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

    print('\'%s\'...' %(in_imagesFile))

    if (FORMATINPUTIMAGES=='dicom'):
        # revert images and start from traquea
        in_volImages_array = DICOMreader.getImageArray(in_imagesFile)[::-1]
        in_volMasks_array  = DICOMreader.getImageArray(in_masksFile) [::-1]
    elif (FORMATINPUTIMAGES=='nifti'):
        # revert images and start from traquea
        in_volImages_array = NIFTIreader.getImageArray(in_imagesFile)[::-1]
        in_volMasks_array  = NIFTIreader.getImageArray(in_masksFile) [::-1]


    totalVols   = in_volImages_array.shape[0] // IMAGES_DEPTHZ
    totalSlices = totalVols * IMAGES_DEPTHZ

    if (CROPPINGIMAGES and AUGMENTDATATWOSIDES):
        # augment data by including separately two lungs sides
        out_volImages_array = np.ndarray([2*totalVols, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATIMAGEDATA)
        out_volMasks_array  = np.ndarray([2*totalVols, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATMASKDATA)
    else:
        out_volImages_array = np.ndarray([totalVols, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATIMAGEDATA)
        out_volMasks_array  = np.ndarray([totalVols, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATMASKDATA)


    if (CROPPINGIMAGES):
        # Cropping slices of CT volumes
        countVol   = 0
        for iVol in range(totalVols):

            batch_images = in_volImages_array[iVol*IMAGES_DEPTHZ:(iVol + 1)*IMAGES_DEPTHZ, :,:]
            batch_masks  = in_volMasks_array [iVol*IMAGES_DEPTHZ:(iVol + 1)*IMAGES_DEPTHZ, :,:]

            # Turn the masks to binary labels:
            batch_masks = np.where(batch_masks != 0, 1, 0)

            if (CHECKMASKSOUTBOUNDBOX):
                # check for binary masks outside bounding box in slice
                if (checkMaskOutBoundBox(batch_masks, CROPPATHTWOSIDES)):
                    message = "Found binary masks outside Bounding-Box in batch volume"
                    CatchErrorException(message)

            cropBatch_images = batch_images[:, CROPPATCH[0][0]:CROPPATCH[0][1],
                                               CROPPATCH[1][0]:CROPPATCH[1][1]]
            cropBatch_masks  = batch_masks[:, CROPPATCH[0][0]:CROPPATCH[0][1],
                                              CROPPATCH[1][0]:CROPPATCH[1][1]]

            out_volImages_array[countVol] = np.asarray(cropBatch_images, dtype=FORMATIMAGEDATA)
            out_volMasks_array [countVol] = np.asarray(cropBatch_masks,  dtype=FORMATMASKDATA)

            countVol += 1

            if (AUGMENTDATATWOSIDES):
                # doubling data by including separately both lung sides

                cropBatch_images = batch_images[:, CROPPATCHMIRROR[0][0]:CROPPATCHMIRROR[0][1],
                                                   CROPPATCHMIRROR[1][0]:CROPPATCHMIRROR[1][1]]
                cropBatch_masks  = batch_masks[:, CROPPATCHMIRROR[0][0]:CROPPATCHMIRROR[0][1],
                                                  CROPPATCHMIRROR[1][0]:CROPPATCHMIRROR[1][1]]

                out_volImages_array[countVol] = np.asarray(cropBatch_images, dtype=FORMATIMAGEDATA)
                out_volMasks_array [countVol] = np.asarray(cropBatch_masks,  dtype=FORMATMASKDATA)

                countVol += 1
        # endfor
    else:
        for iVol in range(totalVols):

            batch_images = in_volImages_array[iVol*IMAGES_DEPTHZ:(iVol + 1)*IMAGES_DEPTHZ, :,:]
            batch_masks  = in_volMasks_array [iVol*IMAGES_DEPTHZ:(iVol + 1)*IMAGES_DEPTHZ, :,:]

            out_volImages_array[iVol] = np.asarray(batch_images[iVol*IMAGES_DEPTHZ:(iVol + 1)*IMAGES_DEPTHZ, :,:], dtype=FORMATIMAGEDATA)
            out_volMasks_array [iVol] = np.asarray(batch_masks [iVol*IMAGES_DEPTHZ:(iVol + 1)*IMAGES_DEPTHZ, :,:], dtype=FORMATMASKDATA )
        #endfor


    stringinfo = "_".join(str(i) for i in list(out_volImages_array.shape))

    out_imagesFile = os.path.join(ProcVolsDataPath, 'volsImages-%0.2i_dim'%(i)+stringinfo+'.npy')
    out_masksFile  = os.path.join(ProcVolsDataPath, 'volsMasks-%0.2i_dim'%(i)+stringinfo+'.npy')

    np.save(out_imagesFile, out_volImages_array)
    np.save(out_masksFile,  out_volMasks_array )
#endfor