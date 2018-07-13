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
from CommonUtil.FunctionsUtil import *
from CommonUtil.WorkDirsManager import WorkDirsManager
import numpy as np
import scipy.io
import math


def getCenterlineLength(centerline):
    length  = 0
    num_points = centerline.shape[0]
    for i in range(num_points-1):
        dist = centerline[i+1] - centerline[i]
        length += math.sqrt(dist[0]*dist[0] + dist[1]*dist[1] + dist[2]*dist[2])
    return length


def main():

    workDirsManager           = WorkDirsManager(BASEDIR)
    RawImagesFilesPath        = workDirsManager.getNameNewPath(workDirsManager.getNameTestingDataPath(), 'RawImages')
    RawMasksFilesPath         = workDirsManager.getNameNewPath(workDirsManager.getNameTestingDataPath(), 'RawMasks')
    RawCenterlinesFilesPath   = workDirsManager.getNameNewPath(workDirsManager.getNameTestingDataPath(), 'RawCenterlines')
    MatLabCenterlinesFilesPath= workDirsManager.getNameNewPath(workDirsManager.getNameTestingDataPath(), 'MatLabCenterlines')

    # Get the file list:
    listImagesFiles      = findFilesDir(RawImagesFilesPath + '/av*.dcm')
    listMasksFiles       = findFilesDir(RawMasksFilesPath + '/av*seg.dcm' )
    listCenterlinesFiles = findFilesDir(MatLabCenterlinesFilesPath + '/av*centerlines.mat')


    for images_file, masks_file, centerlinesFile in zip(listImagesFiles, listMasksFiles, listCenterlinesFiles):

        print('\'%s\'...' % (images_file))

        voxel_size = DICOMreader.getImageVoxelSize(images_file)

        raw_airways_centerlines = scipy.io.loadmat(centerlinesFile)['airway'][0]

        list_centerlines = []
        for centerline in raw_airways_centerlines:
            for i, point in enumerate(centerline[5]):
                centerline[5][i] = point*voxel_size
            list_centerlines.append(centerline[5])
        #endfor

        num_airways = len(list_centerlines)

        total_length = 0.0
        for centerline in list_centerlines:
            total_length += getCenterlineLength(centerline)
        #endfor

        print('Number of airways: %s, with total length: %s'%(num_airways, total_length))
    #endfor


if __name__ == "__main__":
    main()