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

TYPEDATA = 'training'

FORMATINPUTIMAGES = 'dicom'


#MAIN
workDirsManager = WorkDirsManager(BASEDIR)
BaseDataPath    = workDirsManager.getNameDataPath(TYPEDATA)
RawImagesPath   = workDirsManager.getNameRawImagesDataPath(TYPEDATA)
RawMasksPath    = workDirsManager.getNameRawMasksDataPath(TYPEDATA)
ProcVolsDataPath= workDirsManager.getNameNewPath(BaseDataPath, 'VolsData')


# Get the file list:
listImagesFiles = sorted(glob(RawImagesPath + '/*.dcm'))
listMasksFiles  = sorted(glob(RawMasksPath  + '/*.dcm'))

nbImagesFiles = len(listImagesFiles)
nbMasksFiles  = len(listMasksFiles)

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


    # Turn the masks to binary labels:
    in_volMasks_array = np.where(in_volMasks_array != 0, 1, 0)

    out_volImages_array = np.ndarray(in_volImages_array.shape, dtype=FORMATIMAGEDATA)
    out_volMasks_array  = np.ndarray(in_volMasks_array.shape,  dtype=FORMATMASKDATA)

    out_volImages_array[:,:,:] = in_volImages_array
    out_volMasks_array [:,:,:] = in_volMasks_array


    stringinfo = "_".join(str(i) for i in list(out_volImages_array.shape))

    out_imagesFile = os.path.join(ProcVolsDataPath, 'volsImages-%0.2i_dim'%(i)+stringinfo+'.npy')
    out_masksFile  = os.path.join(ProcVolsDataPath, 'volsMasks-%0.2i_dim'%(i)+stringinfo+'.npy')

    np.save(out_imagesFile, out_volImages_array)
    np.save(out_masksFile,  out_volMasks_array )
#endfor