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
from CommonUtil.DICOMreader import DICOMreader
from CommonUtil.ErrorMessages import *
from CommonUtil.WorkDirsManager import WorkDirsManager
from glob import glob
import numpy as np
import os

TYPEDATA = 'testing'


#MAIN
workDirsManager = WorkDirsManager(BASEDIR)

DataPath        = workDirsManager.getNameDataPath(TYPEDATA)
ImagesPath      = workDirsManager.getNameRawImagesDataPath(TYPEDATA)
GroundTruthPath = workDirsManager.getNameRawGroundTruthDataPath(TYPEDATA)

if( not os.path.isdir(ImagesPath) ):
    message = "directory \'%s\' does not exist" % (ImagesPath)
    CatchErrorException(message)

if( not os.path.isdir(GroundTruthPath) ):
    message = "directory \'%s\' does not exist" % (GroundTruthPath)
    CatchErrorException(message)


# Get the file list:
listImageFiles      = sorted(glob(ImagesPath + '*.dcm'))
listGroundTruthFiles= sorted(glob(GroundTruthPath  + '*.dcm'))


boundBoxAllFiles = [(1000, -1000),  # (row_min, row_max)
                    (1000, -1000)]  # (row_min, row_max)

for imageFile, groundTruthFile in zip(listImageFiles, listGroundTruthFiles):

    print('\'%s\'...' %(imageFile))

    sizeImage       = DICOMreader.getImageSize(imageFile)
    sizeGroundTruth = DICOMreader.getImageSize(groundTruthFile)

    if (sizeImage != sizeGroundTruth):
        message = "size of Images not equal to size of GroundTruth"
        CatchErrorException(message)

    groundTruth_array = DICOMreader.loadImage(groundTruthFile)

    boundBox = [(1000, -1000),  # (row_min, row_max)
                (1000, -1000)]  # (row_min, row_max)

    for i, slice_groundTruth in enumerate(groundTruth_array):

        # Find out where there are active masks: segmentations. Elsewhere is not interesting for us
        indexesActiveMask = np.argwhere(slice_groundTruth != 0)

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
