#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.PlotsManager import *
from CommonUtil.WorkDirsManager import *
from glob import glob
import numpy as np
import os


#MAIN
workDirsManager     = WorkDirsManager(BASEDIR)
InputDICOMfilesPath = workDirsManager.getNameNewPath(workDirsManager.getNameTestingDataPath(), 'RawCenterlines')
OutputDICOMfilesPath= workDirsManager.getNameNewPath(workDirsManager.getNameTestingDataPath(), 'RawCenterlines')


# LOADING DATA
# ----------------------------------------------
print('-' * 30)
print('Loading data...')
print('-' * 30)

listInputDICOMFiles = sorted(glob(InputDICOMfilesPath + '/av*.dcm'))

for inputDICOMfile in listInputDICOMFiles:

    print('\'%s\'...' % (inputDICOMfile))

    image_array = DICOMreader.getImageArray(inputDICOMfile)

    outputNIFTIfile = os.path.join(OutputDICOMfilesPath, os.path.basename(inputDICOMfile).replace('.dcm','.nii'))

    # Important: in nifty format, the axes are reversed
    NIFTIreader.writeImageArray(outputNIFTIfile, np.swapaxes(image_array, 0, 2))
#endfor