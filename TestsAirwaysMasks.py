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



#MAIN
workDirsManager  = WorkDirsManager(BASEDIR)
BaseDataPath     = workDirsManager.getNameDataPath(TYPEDATA)
MasksPath        = workDirsManager.getNameNewPath(BaseDataPath, 'RawMasks')
ProcVolsDataPath = workDirsManager.getNameNewPath(BaseDataPath, 'ProcMasks')


# Get the file list:
listMasksFiles   = findFilesDir(MasksPath + '/*.dcm')[0:1]
nbMasksFiles     = len(listMasksFiles)


for i, masksFile in enumerate(listMasksFiles):

    print('\'%s\'...' %(masksFile))

    masks_array = FileReader.getImageArray(masksFile)

    indexes_segs_masks = np.unique(masks_array)

    # remove background mask from list
    indexes_segs_masks = np.delete(indexes_segs_masks, 0)

    for index in indexes_segs_masks:

        print index

        mask_uniquesegs = np.where(masks_array == index, 1, 0).astype(dtype=masks_array.dtype)

        nameoutfile = joinpathnames(ProcVolsDataPath, filenamenoextension(masksFile)+'_seg%0.2i.nii'%(index))

        FileReader.writeImageArray(nameoutfile, mask_uniquesegs)
    #endfor
#endfor