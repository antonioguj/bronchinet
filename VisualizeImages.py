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
from CommonUtil.FunctionsUtil import *
from CommonUtil.PlotsManager import *
from CommonUtil.WorkDirsManager import *



# MAIN
workDirsManager = WorkDirsManager(BASEDIR)
DataPath        = workDirsManager.getNameNewPath(workDirsManager.getNameDataPath(TYPEDATA), 'ProcVolsData')
OutFilesPath    = workDirsManager.getNameNewPath(workDirsManager.getNameDataPath(TYPEDATA), 'VisualCases')

# Get the file list:
listImagesFiles = findFilesDir(DataPath + '/volsImages*.npy')
listMasksFiles  = findFilesDir(DataPath + '/volsMasks*.npy')

for imageFile, maskFile in zip(listImagesFiles, listMasksFiles):

    images_array = np.load(imageFile).astype(FORMATIMAGEDATA)
    masks_array  = np.load(maskFile) .astype(FORMATMASKDATA )

    print('Number training images: %s' %(images_array.shape[0]))

    PlotsManager.saveplot_image_mask_3D(OutFilesPath, images_array, masks_array)