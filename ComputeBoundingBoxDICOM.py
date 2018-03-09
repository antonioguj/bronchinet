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
from CommonUtil.FileReaders import DICOMreader
from CommonUtil.ErrorMessages import *
from CommonUtil.WorkDirsManager import WorkDirsManager
from glob import glob
import numpy as np
import os

TYPEDATA = 'testing'


#MAIN
workDirsManager= WorkDirsManager(BASEDIR)
DataPath       = workDirsManager.getNameDataPath(TYPEDATA)
ImagesPath     = workDirsManager.getNameRawImagesDataPath(TYPEDATA)
MasksPath      = workDirsManager.getNameRawMasksDataPath(TYPEDATA)

# Get the file list:
listImageFiles = sorted(glob(ImagesPath + '/*.dcm'))
listMaskFiles  = sorted(glob(MasksPath  + '/*.dcm'))


boundBoxAllFiles = [(1000, -1000),  # (row_min, row_max)
                    (1000, -1000)]  # (row_min, row_max)

for imageFile, maskFile in zip(listImageFiles, listMaskFiles):

    print('\'%s\'...' %(imageFile))

    sizeImage = DICOMreader.getImageSize(imageFile)
    sizeMask  = DICOMreader.getImageSize(maskFile)

    if (sizeImage != sizeMask):
        message = "size of Images not equal to size of Mask"
        CatchErrorException(message)

    mask_array = DICOMreader.loadImage(maskFile)

    boundBox = [(1000, -1000),  # (row_min, row_max)
                (1000, -1000)]  # (row_min, row_max)

    for i, slice_mask in enumerate(mask_array):

        # Find out where there are active masks: segmentations. Elsewhere is not interesting for us
        indexesActiveMask = np.argwhere(slice_mask != 0)

        if( len(indexesActiveMask) ):

            boundBoxSlice = [(min(indexesActiveMask[:, 0]), max(indexesActiveMask[:, 0])),
                             (min(indexesActiveMask[:, 1]), max(indexesActiveMask[:, 1]))]

            boundBox = [(min(boundBox[0][0], boundBoxSlice[0][0]), max(boundBox[0][1], boundBoxSlice[0][1])),
                        (min(boundBox[1][0], boundBoxSlice[1][0]), max(boundBox[1][1], boundBoxSlice[1][1]))]
    #endfor

    print('Bounding Box: %s' % (boundBox))

    boundBoxAllFiles = [(min(boundBoxAllFiles[0][0], boundBox[0][0]), max(boundBoxAllFiles[0][1], boundBox[0][1])),
                        (min(boundBoxAllFiles[1][0], boundBox[1][0]), max(boundBoxAllFiles[1][1], boundBox[1][1]))]
#endfor

print('Bounding Box All Files: %s' %(boundBoxAllFiles))
