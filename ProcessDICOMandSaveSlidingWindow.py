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

(PROP_X, PROP_Y, PROP_Z) = (0.5, 0.5, 0.5)  # PROP_OVERLAP

if (TYPEDATA=='validation' or
    TYPEDATA=='testing'):
    # no overlapping for validation or testing:
    # no need for data augmentation
    (PROP_X, PROP_Y, PROP_Z) = (0.0, 0.0, 0.0)


#MAIN
workDirsManager = WorkDirsManager(BASEDIR)
BaseDataPath    = workDirsManager.getNameDataPath(TYPEDATA)
RawImagesPath   = workDirsManager.getNameRawImagesDataPath(TYPEDATA)
RawMasksPath    = workDirsManager.getNameRawMasksDataPath(TYPEDATA)
ProcVolsDataPath= workDirsManager.getNameNewPath(BaseDataPath, 'SlidePatchsVolsData')


# Get the file list:
listImagesFiles = sorted(glob(RawImagesPath + '/*.dcm'))
listMasksFiles  = sorted(glob(RawMasksPath  + '/*.dcm'))

nbImagesFiles = len(listImagesFiles)
nbMasksFiles  = len(listMasksFiles)

nbFiles = len(listImagesFiles)

print('-' * 30)
print('Loading CT volumes (%s in total), Computing Sliding Window Patches, and Saving...' % (nbFiles))
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


    size_volImages_array = in_volImages_array.shape
    size_volMasks_array  = in_volMasks_array.shape

    if (size_volImages_array != size_volMasks_array):
        message = "Images array of different size to Masks array (\'%s\',\'%s\')" % (size_volImages_array, size_volMasks_array)
        CatchErrorException(message)

    (size_img_Z, size_img_X, size_img_Y) = size_volImages_array

    # compute num batches in 3 directions
    num_cropimgs_X = int(np.floor((size_img_X - PROP_X*IMAGES_HEIGHT) /(1 - PROP_X)/IMAGES_HEIGHT))
    num_cropimgs_Y = int(np.floor((size_img_Y - PROP_Y*IMAGES_WIDTH ) /(1 - PROP_Y)/IMAGES_WIDTH ))
    num_cropimgs_Z = int(np.floor((size_img_Z - PROP_Z*IMAGES_DEPTHZ) /(1 - PROP_Z)/IMAGES_DEPTHZ))
    num_cropimages = num_cropimgs_X * num_cropimgs_Y * num_cropimgs_Z

    print('Sliding Window: build num images: %s; in X_Y_Z dirs: (%s,%s,%s)'%(num_cropimages, num_cropimgs_X, num_cropimgs_Y, num_cropimgs_Z))


    out_volImages_array = np.ndarray([num_cropimages, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATIMAGEDATA)
    out_volMasks_array  = np.ndarray([num_cropimages, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATMASKDATA )

    for idx_batch in range(num_cropimages):

        idx_Z_batch = idx_batch // (num_cropimgs_X * num_cropimgs_Y)
        idx_batch_XY= idx_batch % (num_cropimgs_X * num_cropimgs_Y)
        idx_Y_batch = idx_batch_XY // num_cropimgs_X
        idx_X_batch = idx_batch_XY % num_cropimgs_X

        # Compute bounding-box for batch (idxX, idxY, idxZ)
        x_left = int(idx_X_batch * (1.0 - PROP_X) * IMAGES_HEIGHT)
        x_right= x_left + IMAGES_HEIGHT
        y_down = int(idx_Y_batch * (1.0 - PROP_Y) * IMAGES_WIDTH)
        y_up   = y_down + IMAGES_WIDTH
        z_back = int(idx_Z_batch * (1.0 - PROP_Z) * IMAGES_DEPTHZ)
        z_front= z_back + IMAGES_DEPTHZ

        out_volImages_array[idx_batch] = np.asarray(in_volImages_array[z_back:z_front, x_left:x_right, y_down:y_up], dtype=FORMATIMAGEDATA)
        out_volMasks_array [idx_batch] = np.asarray(in_volMasks_array [z_back:z_front, x_left:x_right, y_down:y_up], dtype=FORMATMASKDATA )
    #endfor


    stringinfo = "_".join(str(i) for i in list(out_volImages_array.shape))

    out_imagesFile = os.path.join(ProcVolsDataPath, 'volsImages-%0.2i_dim'%(i)+stringinfo+'.npy')
    out_masksFile  = os.path.join(ProcVolsDataPath, 'volsMasks-%0.2i_dim'%(i)+stringinfo+'.npy')

    np.save(out_imagesFile, out_volImages_array)
    np.save(out_masksFile,  out_volMasks_array )
#endfor