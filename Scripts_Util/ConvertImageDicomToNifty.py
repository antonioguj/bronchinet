#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

import argparse

from DataLoaders.FileReaders import *
from OperationImages.OperationMasks import *


def main(args):
    # ---------- SETTINGS ----------
    InputPath = args.inputdir
    OutputPath = args.outputdir
    namesOutputFiles = lambda in_name: filenamenoextension(in_name) + '.nii.gz'
    # ---------- SETTINGS ----------

    listInputFiles = findFilesDirAndCheck(InputPath)
    makedir(OutputPath)


    for in_file in listInputFiles:
        print("\nInput: \'%s\'..." % (basename(in_file)))

        inout_array = FileReader.getImageArray(in_file)

        if (args.isBinaryImage):
            print("Convert image to binary masks (0, 1)...")
            inout_array = OperationBinaryMasks.process_masks(inout_array)

        out_file = joinpathnames(OutputPath, namesOutputFiles(in_file))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_file), str(inout_array.shape)))

        FileReader.writeImageArray(out_file, inout_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str)
    parser.add_argument('outputdir', type=str)
    parser.add_argument('--isBinaryImage', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)