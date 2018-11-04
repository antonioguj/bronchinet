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
from Preprocessing.BoundingBoxMasks import *
from Preprocessing.OperationsImages import *
import argparse



def main(args):

    # ---------- SETTINGS ----------
    nameInputImagesRelPath  = args.inputdir
    nameReferImagesRelPath  = 'ProcImages'
    nameOutputImagesRelPath = args.outputdir

    # Get the file list:
    nameInputImagesFiles = '*.nii.gz'
    nameReferImagesFiles = '*.nii.gz'

    # template search files
    tempSearchInputFiles = 'vol[0-9][0-9]_*'

    nameBoundingBoxesImages = 'found_boundBoxes_Original.npy'

    nameOutputImagesFiles = lambda in_name: filenamenoextension(in_name) + '.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager  = WorkDirsManager(args.basedir)
    BaseDataPath     = workDirsManager.getNameBaseDataPath()
    InputImagesPath  = workDirsManager.getNameExistPath(args.basedir, nameInputImagesRelPath)
    ReferImagesPath  = workDirsManager.getNameExistPath(BaseDataPath, nameReferImagesRelPath)
    OutputImagesPath = workDirsManager.getNameNewPath  (args.basedir, nameOutputImagesRelPath)

    listInputImagesFiles = findFilesDir(InputImagesPath, nameInputImagesFiles)
    listReferImagesFiles = findFilesDir(ReferImagesPath, nameReferImagesFiles)

    nbInputImagesFiles = len(listInputImagesFiles)
    nbReferImagesFiles = len(listReferImagesFiles)


    namefile_dict = joinpathnames(BaseDataPath, nameBoundingBoxesImages)
    dict_boundingBoxes = readDictionary(namefile_dict)



    for i, images_file in enumerate(listInputImagesFiles):

        print('file: \'%s\'...' % (images_file))

        name_prefix_case = getExtractSubstringPattern(basename(images_file),
                                                      tempSearchInputFiles)

        for iterfile in listReferImagesFiles:
            if name_prefix_case in iterfile:
                refer_images_file = iterfile
        # endfor
        print("assigned to '%s'..." % (basename(refer_images_file)))

        bounding_box = dict_boundingBoxes[filenamenoextension(refer_images_file)]


        images_full_array = FileReader.getImageArray(images_file)

        print("dimensions of input image: %s..." % (str(images_full_array.shape)))

        # 1 step: crop image
        images_cropped_array = CropImages.compute3D(images_full_array, bounding_box)

        # 2 step: invert image
        images_cropped_array = FlippingImages.compute(images_cropped_array, axis=0)

        print("dimensions of cropped image: %s..." % (str(images_cropped_array.shape)))


        out_images_filename = joinpathnames(OutputImagesPath, nameOutputImagesFiles(images_file))

        print("Saving image: %s..." %(out_images_filename))

        FileReader.writeImageArray(out_images_filename, images_cropped_array)
    #endfor



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--inputdir')
    parser.add_argument('--outputdir')
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