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
from Preprocessing.OperationsImages import *
import argparse



def main(args):

    # ---------- SETTINGS ----------
    # Get the file list:
    nameInputFiles  = '*.dcm'
    nameOutputFiles = '.nii.gz'

    nameOutputFiles = lambda in_name: filenamenoextension(in_name).replace('surface0','lumen').replace('surface1','outerwall') + '.nii.gz'
    # ---------- SETTINGS ----------


    InputPath  = WorkDirsManager.getNameExistPath(args.basedir, args.inputdir )
    OutputPath = WorkDirsManager.getNameNewPath  (args.basedir, args.outputdir)

    listInputFiles = findFilesDir(InputPath, nameInputFiles)

    nbInputFiles = len(listInputFiles)

    # Run checkers
    if (nbInputFiles == 0):
        message = "0 Images found in dir \'%s\'" %(InputPath)
        CatchErrorException(message)


    for i, input_file in enumerate(listInputFiles):

        print('\'%s\'...' %(input_file))

        images_array = FileReader.getImageArray(input_file)

        if (args.invertImageAxial):
            images_array = FlippingImages.compute(images_array, axis=0)


        print("Saving images in nifty '.nii' format of final dimensions: %s..." %(str(images_array.shape)))

        out_images_filename = joinpathnames(OutputPath, nameOutputFiles(input_file))

        FileReader.writeImageArray(out_images_filename, images_array.astype(FORMATIMAGEDATA))
    #endfor



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--inputdir', type=str)
    parser.add_argument('--outputdir', type=str)
    parser.add_argument('--invertImageAxial', type=str2bool, default=INVERTIMAGEAXIAL)
    args = parser.parse_args()

    if not args.inputdir:
        message = 'Please input a valid input directory'
        CatchErrorException(message)
    if not args.outputdir:
        message = 'Output directory not indicated. Assume same as input directory'
        args.outputdir = args.inputdir
        CatchWarningException(message)

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)