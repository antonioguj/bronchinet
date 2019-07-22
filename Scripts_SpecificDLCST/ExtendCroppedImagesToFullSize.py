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
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameInputImagesRelPath     = args.inputdir
    nameInputReferFilesRelPath = 'Images_Full'
    nameOutputImagesRelPath    = args.outputdir
    nameInputImagesFiles       = '*.nii.gz'
    nameInputReferFiles        = '*.nii.gz'
    nameBoundingBoxes          = 'found_boundBoxes_original.npy'
    nameOutputImagesFiles      = lambda in_name: filenamenoextension(in_name) + '.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager     = WorkDirsManager(args.datadir)
    InputImagesPath     = workDirsManager.getNameExistPath(nameInputImagesRelPath)
    InputReferFilesPath = workDirsManager.getNameExistPath(nameInputReferFilesRelPath)
    OutputImagesPath    = workDirsManager.getNameNewPath  (nameOutputImagesRelPath)

    listInputImagesFiles = findFilesDirAndCheck(InputImagesPath,     nameInputImagesFiles)
    listInputReferFiles  = findFilesDirAndCheck(InputReferFilesPath, nameInputReferFiles)

    dict_bounding_boxes = readDictionary(joinpathnames(BaseDataPath, nameBoundingBoxes))



    for i, in_image_file in enumerate(listInputImagesFiles):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))

        in_refer_file = findFileWithSamePrefix(basename(in_image_file), listInputReferFiles,
                                               prefix_pattern='vol[0-9][0-9]_')
        print("Reference file: \'%s\'..." % (basename(in_refer_file)))
        bounding_box = dict_bounding_boxes[filenamenoextension(in_refer_file)]


        crop_image_array = FileReader.getImageArray(in_image_file)
        print("Input cropped image size: \'%s\'..." %(str(crop_image_array.shape)))

        # 1 step: invert image
        crop_image_array = FlippingImages.compute(crop_image_array, axis=0)
        # 2 step: extend image
        full_image_shape = FileReader.getImageSize(in_refer_file)
        full_image_array = ExtendImages.compute3D(crop_image_array, bounding_box, full_image_shape)
        print("Output full image size: \'%s\'..." % (str(full_image_array.shape)))

        out_image_file = joinpathnames(OutputImagesPath, nameOutputImagesFiles(in_image_file))
        print("Output: \'%s\', of dims \'%s\'..." %(basename(out_image_file), str(full_image_array.shape)))

        FileReader.writeImageArray(out_image_file, full_image_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=DATADIR)
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