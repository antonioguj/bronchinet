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
from CommonUtil.FileReaders import *
from CommonUtil.FunctionsUtil import *
from CommonUtil.WorkDirsManager import *
import argparse



# ---------- SETTINGS ----------
inputdir  = '/home/antonio/Data/DLCST_Proc/AirwaysDistTrans_Full/AirwaysDistTrans_Normalize'
outputdir = '/home/antonio/Data/DLCST_Proc/AirwaysDistTrans_Full/AirwaysDistTrans_Normalize_Exponential'

# Get the file list:
nameInputFiles  = '*.nii.gz'

nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_expon.nii.gz'

def fun_math_oper(images_array):
    return (np.exp(images_array)-1.0)/(np.exp(1)-1.0)
    #return np.power(images_array, 10)
# ---------- SETTINGS ----------


InputPath  = WorkDirsManager.getNameExistPath(inputdir )
OutputPath = WorkDirsManager.getNameNewPath  (outputdir)

listInputFiles = findFilesDir(InputPath, nameInputFiles)

nbInputFiles = len(listInputFiles)

# Run checkers
if (nbInputFiles == 0):
    message = "0 Images found in dir \'%s\'" %(InputPath)
    CatchErrorException(message)


for i, input_file in enumerate(listInputFiles):

    print('\'%s\'...' %(input_file))

    images_array = FileReader.getImageArray(input_file)

    out_images_array = fun_math_oper(images_array)

    out_images_filename = joinpathnames(OutputPath, nameOutputFiles(input_file))

    FileReader.writeImageArray(out_images_filename, out_images_array)
#endfor