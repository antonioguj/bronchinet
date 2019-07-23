#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.Constants import *
from Common.WorkDirsManager import *
from DataLoaders.FileReaders import *
from Preprocessing.OperationImages import *



# ---------- SETTINGS ----------
nameInputRelPath = '/home/antonio/Data/iBEST_Raw/Annotations'
nameOutputRelPath = '/home/antonio/Data/iBEST_Raw/Annotations_nifty'

nameInputAxialFiles = '*axialCL.dcm'
nameInputCoronalFiles = '*coronalCL.dcm'
nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '.nii.gz'
nameOutput2Files = lambda in_name: filenamenoextension(in_name) + '_reshaped.nii.gz'
# ---------- SETTINGS ----------


workDirsManager = WorkDirsManager(DATADIR)
InputPath       = WorkDirsManager.getNameExistPath(nameInputRelPath)
OutputPath      = WorkDirsManager.getNameNewPath  (nameOutputRelPath)

listInputAxialFiles   = findFilesDirAndCheck(InputPath, nameInputAxialFiles)
listInputCoronalFiles = findFilesDirAndCheck(InputPath, nameInputCoronalFiles)



for i, (in_axial_file, in_coronal_file) in enumerate(zip(listInputAxialFiles, listInputCoronalFiles)):
    print("\nInput: \'%s\'..." % (basename(in_axial_file)))
    print("And: \'%s\'..." % (basename(in_coronal_file)))

    in_axialCL_array = FileReader.getImageArray(in_axial_file)
    in_coronalCL_array = FileReader.getImageArray(in_coronal_file)

    coronalCL_reshaped_array = FlippingImages.compute(in_coronalCL_array.swapaxes(1,0), axis=0)

    diff_two_arrays = np.abs(in_axialCL_array - coronalCL_reshaped_array)

    unique_values = np.unique(diff_two_arrays)
    if unique_values != [0]:
        print("CHECK: axial and Coronal views are different...")
    else:
        print("GOOD: axial and Coronal views are same...")


    out_axial_filename = joinpathnames(OutputPath, nameOutputFiles(in_axial_file))
    out_coronal_filename = joinpathnames(OutputPath, nameOutputFiles(in_coronal_file))
    out_coronal_reshaped_filename = joinpathnames(OutputPath, nameOutput2Files(in_coronal_file))

    FileReader.writeImageArray(out_axial_filename, in_axialCL_array)
    FileReader.writeImageArray(out_coronal_filename, in_coronalCL_array)
    FileReader.writeImageArray(out_coronal_reshaped_filename, coronalCL_reshaped_array)
#endfor