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

MAXSLICESINIMAGESFILE = 3000

FORMATINPUTIMAGES = 'dicom'
CROPPINGIMAGES = True
AUGMENTDATATWOSIDES = True
CHECKMASKSOUTBOUNDBOX = False

SHUFFLEIMAGES = True


def saveImagesAndMasksFiles(slicesImagesData, slicesMasksData, countFiles):

    stringinfo = "_".join(str(i) for i in list(slicesImagesData.shape))

    nameOutImagesFile = os.path.join(ProcSlicesDataPath, 'slicesImages-%0.2i_dim'%(countFiles)+stringinfo+'.npy')
    nameOutMasksFile  = os.path.join(ProcSlicesDataPath, 'slicesMasks-%0.2i_dim'%(countFiles)+stringinfo+'.npy')

    print('-' * 30)
    print('Saving file (%s)...' %(nameOutImagesFile))
    print('-' * 30)

    nbImages = slicesImagesData.shape[0]
    if SHUFFLEIMAGES:
        # Generate random indexes in range (0,num_files) to shuffle input data of DNN
        randIndexes = np.random.choice(nbImages, size=nbImages, replace=False)

        np.save(nameOutImagesFile, slicesImagesData[randIndexes[:]])
        np.save(nameOutMasksFile,  slicesMasksData [randIndexes[:]])
    else:
        # Do not shuffle data
        np.save(nameOutImagesFile, slicesImagesData)
        np.save(nameOutMasksFile,  slicesMasksData)


def checkMaskOutBoundBox(image_array, croppatch):

    image_array[croppatch[0][0]:croppatch[0][1],
                croppatch[1][0]:croppatch[1][1]] = 0
    return (np.count_nonzero(image_array) != 0)


#MAIN
workDirsManager   = WorkDirsManager(BASEDIR)
BaseDataPath      = workDirsManager.getNameDataPath(TYPEDATA)
RawImagesPath     = workDirsManager.getNameRawImagesDataPath(TYPEDATA)
RawMasksPath      = workDirsManager.getNameRawMasksDataPath(TYPEDATA)
ProcSlicesDataPath= workDirsManager.getNameNewPath(BaseDataPath, 'ProcSlicesData')


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
    message = "nb Images files %i not equal to nb Masks Files %i" %(nbImagesFiles,nbMasksFiles)
    CatchErrorException(message)


out_slicesImages_array = np.ndarray([MAXSLICESINIMAGESFILE, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATIMAGEDATA)
out_slicesMasks_array  = np.ndarray([MAXSLICESINIMAGESFILE, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATIMAGEDATA)

print('-' * 30)
print('Loading files (%s in total), and saving files with maximum images (%s)...' %(nbImagesFiles, MAXSLICESINIMAGESFILE))
print('-' * 30)

countFiles  = 1
countSlices = 0

for imagesFile, masksFile in zip(listImagesFiles, listMasksFiles):

    print('File (%s)...' % (imagesFile))

    if (FORMATINPUTIMAGES=='dicom'):
        # revert images and start from traquea
        in_volImage_array = DICOMreader.getImageArray(imagesFile)[::-1]
        in_volMasks_array = DICOMreader.getImageArray(masksFile) [::-1]
    elif (FORMATINPUTIMAGES=='nifti'):
        # revert images and start from traquea
        in_volImage_array = NIFTIreader.getImageArray(imagesFile)[::-1]
        in_volMasks_array = NIFTIreader.getImageArray(masksFile) [::-1]


    # Save images and ground-truth slices in a global list:
    if (CROPPINGIMAGES):
        for slice_image, slice_masks in zip(in_volImage_array, in_volMasks_array):

            if (len(np.argwhere(slice_masks))):
                # Extract only slices with binary masks

                # Turn the mask images to binary images:
                slice_masks = np.where(slice_masks != 0, 1, 0)

                if (CHECKMASKSOUTBOUNDBOX):
                    # check for binary masks outside bounding box in slice
                    if (checkMaskOutBoundBox(slice_masks, CROPPATHTWOSIDES)):
                        message = "Found masks outside Bounding-Box in slice"
                        CatchErrorException(message)

                cropslice_image = slice_image[CROPPATCH[0][0]:CROPPATCH[0][1],
                                              CROPPATCH[1][0]:CROPPATCH[1][1]]
                cropslice_masks = slice_masks[CROPPATCH[0][0]:CROPPATCH[0][1],
                                              CROPPATCH[1][0]:CROPPATCH[1][1]]

                out_slicesImages_array[countSlices] = np.asarray(cropslice_image, dtype=FORMATIMAGEDATA)
                out_slicesMasks_array [countSlices] = np.asarray(cropslice_masks, dtype=FORMATMASKDATA)

                countSlices += 1

                if (AUGMENTDATATWOSIDES):
                    # doubling data by including separately both lung sides

                    cropslice_image = slice_image[CROPPATCHMIRROR[0][0]:CROPPATCHMIRROR[0][1],
                                                  CROPPATCHMIRROR[1][0]:CROPPATCHMIRROR[1][1]]
                    cropslice_masks = slice_masks[CROPPATCHMIRROR[0][0]:CROPPATCHMIRROR[0][1],
                                                  CROPPATCHMIRROR[1][0]:CROPPATCHMIRROR[1][1]]

                    out_slicesImages_array[countSlices] = np.asarray(cropslice_image, dtype=FORMATIMAGEDATA)
                    out_slicesMasks_array [countSlices] = np.asarray(cropslice_masks, dtype=FORMATMASKDATA)

                    countSlices += 1

                if (countSlices > MAXSLICESINIMAGESFILE-1):

                    saveImagesAndMasksFiles(out_slicesImages_array, out_slicesMasks_array, countFiles)
                    #next file
                    countFiles += 1
                    countSlices = 0
        # endfor
    else:
        for slice_image, slice_masks in zip(in_volImage_array, in_volMasks_array):

            # Turn the mask images to binary images:
            slice_masks = np.where(slice_masks != 0, 1, 0)

            out_slicesImages_array[countSlices] = np.asarray(slice_image, dtype=FORMATIMAGEDATA)
            out_slicesMasks_array [countSlices] = np.asarray(slice_masks, dtype=FORMATMASKDATA)

            countSlices += 1

            if (countSlices > MAXSLICESINIMAGESFILE-1):

                saveImagesAndMasksFiles(out_slicesImages_array, out_slicesMasks_array, countFiles)
                #next file
                countFiles += 1
                countSlices = 0
        #endfor
#endfor

#save last file with remaining images
saveImagesAndMasksFiles(out_slicesImages_array[0:countSlices], out_slicesMasks_array[0:countSlices], countFiles)