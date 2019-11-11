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
from Preprocessing.OperationMasks import *
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameInputImagesRelPath  = args.inputmasksdir
    nameOutputImagesRelPath = args.outputcentrelinesdir
    nameInputImagesFiles    = '*.nii.gz'
    nameOutputImagesFiles   = lambda in_name: filenamenoextension(in_name) + '_cenlines.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager  = WorkDirsManager(args.basedir)
    InputImagesPath  = workDirsManager.getNameExistPath(nameInputImagesRelPath)
    OutputImagesPath = workDirsManager.getNameNewPath  (nameOutputImagesRelPath)

    listInputImagesFiles = findFilesDirAndCheck(InputImagesPath, nameInputImagesFiles)


    for i, in_mask_file in enumerate(listInputImagesFiles):
        print("\nInput: \'%s\'..." % (basename(in_mask_file)))

        in_mask_array = FileReader.getImageArray(in_mask_file)

        print("Compute centrelines by thinning the masks...")
        out_centreline_array = ThinningMasks.compute(in_mask_array)

        out_file = joinpathnames(OutputImagesPath, nameOutputImagesFiles(basename(in_mask_file)))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_file), str(out_centreline_array.shape)))

        FileReader.writeImageArray(out_file, out_centreline_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('inputmasksdir', type=str)
    parser.add_argument('outputcentrelinesdir', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
