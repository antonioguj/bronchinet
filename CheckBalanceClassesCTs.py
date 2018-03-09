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


#MAIN
workDirsManager= WorkDirsManager(BASEDIR)
DataPath       = '/home/antonio/testSegmentation/Data/LUVAR/Segmentations/'

# Get the file list:
listMasksFiles = sorted(glob(DataPath + '/*_surface1.dcm'))

sum_values = 0
for maskFile in listMasksFiles:

    print('\'%s\'...' %(maskFile))

    mask_array = DICOMreader.getImageArray(maskFile)

    num_negclass = 0
    num_posclass = 0
    for slice in mask_array:

        num_posclass += np.count_nonzero(slice)
        num_negclass += np.size(slice) - np.count_nonzero(slice)
    #endfor

    ratio_negtoposclass = num_negclass / num_posclass
    sum_values += ratio_negtoposclass

    print('Ratio negative to positive classes: %s' %(ratio_negtoposclass))

#endfor

print('Average value: %s' %(sum_values/len(listMasksFiles)))