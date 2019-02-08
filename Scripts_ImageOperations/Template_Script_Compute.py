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
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameInputRelPath = '<input_dir>'
    nameOutputRelPath = '<output_dir>'

    nameInputFiles = '*.nii.gz'

    def nameOutputFiles(in_name):
        return in_name.replace('<suffix_ini>', '<suffix_end>')
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(BASEDIR)
    BaseDataPath = workDirsManager.getNameBaseDataPath()
    InputPath = workDirsManager.getNameExistPath(BaseDataPath, nameInputRelPath)
    OutputPath = workDirsManager.getNameNewPath(BaseDataPath, nameOutputRelPath)

    listInputFiles = findFilesDirAndCheck(InputPath, nameInputFiles)


    for i, in_file in enumerate(listInputFiles):
        print("\nInput: \'%s\'..." % (basename(in_file)))

        in_array = FileReader.getImageArray(in_file)

        # ...
        # write here the code
        # ...

        out_array = None

        out_file = joinpathnames(OutputPath, nameOutputFiles(basename(in_file)))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_file), str(out_array.shape)))

        FileReader.writeImageArray(out_file, out_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)