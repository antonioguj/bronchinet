#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

#from DataLoaders.FileReaders import *
from Common.FunctionsUtil import *
import argparse
import sys



def main(args):
    # ---------- SETTINGS ----------
    InputPath = args.inputdir
    OutputPath = args.outputdir
    namesInputFiles = '*.hr2'

    bin_convertHR2 = '/home/antonio/Codes/Silas_repository/image-feature-extraction/build/tools/ConvertHR2'

    namesOutputFiles = lambda in_name: filenamenoextension(in_name) + '.nii.gz'
    # ---------- SETTINGS ----------

    listInputFiles = findFilesDirAndCheck(InputPath, namesInputFiles)
    makedir(OutputPath)


    for in_file in listInputFiles:
        print("\nInput: \'%s\'..." % (basename(in_file)))

        out_file = joinpathnames(OutputPath, namesOutputFiles(in_file))
        print("Output: \'%s\'..." % (basename(out_file)))

        command_string = bin_convertHR2 + ' ' + in_file + ' ' + out_file
        print("%s" %(command_string))
        os.system(command_string)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str)
    parser.add_argument('outputdir', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)