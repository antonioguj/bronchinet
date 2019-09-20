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
    MasksPath = args.masksdir
    OutputPath = args.outputdir

    namesInputFiles = '*.nii.gz'
    namesOutputFiles = lambda in_name: filenamenoextension(in_name) + '.nii.gz'
    # ---------- SETTINGS ----------

    listInputFiles = findFilesDirAndCheck(InputPath, namesInputFiles)
    listMasksFiles = findFilesDirAndCheck(MasksPath, namesInputFiles)
    makedir(OutputPath)


    for in_file, in_mask_file in zip(listInputFiles, listMasksFiles):
        print("\nInput: \'%s\'..." % (basename(in_file)))
        print("Masks: \'%s\'..." % (basename(in_mask_file)))

        in_array = FileReader.getImageArray(in_file)
        in_mask_array = FileReader.getImageArray(in_mask_file)

        out_array = in_mask_array * in_array

        out_file = joinpathnames(OutputPath, namesOutputFiles(in_file))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_file), str(out_array.shape)))

        FileReader.writeImageArray(out_file, out_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str)
    parser.add_argument('masksdir', type=str)
    parser.add_argument('outputdir', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)