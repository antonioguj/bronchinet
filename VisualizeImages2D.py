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
from CommonUtil.WorkDirsManager import *
from CommonUtil.PlotsManager import *
from CommonUtil.ErrorMessages import *
import sys

TYPEDATA = 'training'


# MAIN
workDirsManager = WorkDirsManager(BASEDIR)
DataPath        = workDirsManager.getNameNewPath(workDirsManager.getNameTrainingDataPath(), 'ProcSlicesData')

# Get the file list:
listImagesFiles = sorted(glob(DataPath + '/slicesImages*.npy'))
listMasksFiles  = sorted(glob(DataPath + '/slicesMasks*.npy'))

for imageFile, maskFile in zip(listImagesFiles, listMasksFiles):

    images_array = np.load(imageFile).astype(FORMATIMAGEDATA)
    masks_array  = np.load(maskFile) .astype(FORMATMASKDATA )

    print('Number training images: %s' %(images_array.shape[0]))

    PlotsManager.plot_image_mask_2D(images_array, masks_array)