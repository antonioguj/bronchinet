#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from DataLoaders.FileReaders import *
import argparse



def main(args):
    # ---------- SETTINGS ----------
    InputPath = args.inputdir
    OutputPath = args.outputdir

    namesInputFiles = '*.dcm'
    namesOutputFiles = lambda in_name: filenamenoextension(in_name) + '.nii.gz'
    # ---------- SETTINGS ----------

    listInputFiles = findFilesDirAndCheck(InputPath, namesInputFiles)
    makedir(OutputPath)


    for in_file in listInputFiles:
        print("\nInput: \'%s\'..." % (basename(in_file)))

        in_array = DICOMreader.getImageArray(in_file)

        out_file = joinpathnames(OutputPath, namesOutputFiles(in_file))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_file), str(in_array.shape)))

        FileReader.writeImageArray(out_file, in_array)
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