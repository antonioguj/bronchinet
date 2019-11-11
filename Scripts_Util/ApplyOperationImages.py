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
    InputPath  = args.inputdir
    OutputPath = args.outputdir
    nameInputFiles    = '*.nii.gz'
    # ---------- SETTINGS ----------


    listInputFiles  = findFilesDirAndCheck(InputPath, nameInputFiles)
    makedir(OutputPath)


    if args.typeOperation == 'normalize':
        print("Operation: Normalize images...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_normal.nii.gz'

        fun_operation = NormalizeImages.compute3D

    elif args.typeOperation == 'fill_holes':
        print("Operation: Fill holes in images...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_fillholes.nii.gz'

        fun_operation = FillInHolesImages.compute

    elif args.typeOperation == 'thinning':
        print("Operation: Thinning images to centrelines...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_cenlines.nii.gz'

        fun_operation = ThinningMasks.compute

    elif args.typeOperation == 'threshold':
        print("Operation: Threshold images...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_binary.nii.gz'

        fun_operation = ThresholdImages.compute

    elif args.typeOperation == 'power_image':
        print("Operation: Power of image...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_power2.nii.gz'

        def funPowerImage(in_array, order=2):
            return np.power(in_array, order)
        fun_operation = funPowerImage

    elif args.typeOperation == 'exponential_image':
        print("Operation: Exponential of image...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_expon.nii.gz'

        def funPowerImage(in_array):
            return (np.exp(in_array) - 1.0) / (np.exp(1) - 1.0)
        fun_operation = funPowerImage

    else:
        raise Exception("ERROR. type operation '%s' not found... EXIT" %(args.typeOperation))



    for i, in_file in enumerate(listInputFiles):
        print("\nInput: \'%s\'..." % (basename(in_file)))

        in_array = FileReader.getImageArray(in_file)

        out_array = fun_operation(in_array)

        out_file = joinpathnames(OutputPath, nameOutputFiles(basename(in_file)))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_file), str(out_array.shape)))

        FileReader.writeImageArray(out_file, out_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('inputdir', type=str)
    parser.add_argument('outputdir', type=str)
    parser.add_argument('--typeOperation', type=str, default='None')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)