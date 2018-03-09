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
from CommonUtil.FileDataManager import *
from CommonUtil.PlotsManager import *
from CommonUtil.WorkDirsManager import *
from Networks.Metrics import *
from Networks.Networks import *
from glob import glob
import numpy as np
import os


#MAIN
workDirsManager     = WorkDirsManager(BASEDIR)
InputDICOMfilesPath = workDirsManager.getNameNewPath(workDirsManager.getNameTestingDataPath(), 'RawMasks')
OutputDICOMfilesPath= workDirsManager.getNameNewPath(workDirsManager.getNameTestingDataPath(), 'RawMasks')
ModelsPath          = workDirsManager.getNameModelsPath()


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