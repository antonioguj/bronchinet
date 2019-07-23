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
    nameInputReferFilesRelPath = 'RawImages/'
    nameOutputImagesRelPath    = args.outputdir
    nameInputImagesFiles       = '*.dcm'
    nameInputReferFiles        = '*.dcm'
    nameBoundingBoxes          = 'found_boundBoxes_original.npy'
    nameOutputImagesFiles      = lambda in_name: filenamenoextension(in_name) + '.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager     = WorkDirsManager(args.datadir)
    InputImagesPath     = workDirsManager.getNameExistPath(nameInputImagesRelPath)
    InputReferFilesPath = workDirsManager.getNameExistPath(nameInputReferFilesRelPath)
    OutputImagesPath    = workDirsManager.getNameNewPath  (nameOutputImagesRelPath)

    listInputImagesFiles = findFilesDirAndCheck(InputImagesPath,     nameInputImagesFiles)
    listInputReferFiles  = findFilesDirAndCheck(InputReferFilesPath, nameInputReferFiles)

    dict_bounding_boxes = readDictionary(joinpathnames(args.datadir, nameBoundingBoxes))



    for i, in_image_file in enumerate(listInputImagesFiles):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))

        in_refer_file = findFileWithSamePrefix(basename(in_image_file), listInputReferFiles,
                                               prefix_pattern='vol[0-9][0-9]_')
        print("Reference file: \'%s\'..." % (basename(in_refer_file)))
        bounding_box = dict_bounding_boxes[filenamenoextension(in_refer_file)]


        in_fullimage_array = FileReader.getImageArray(in_image_file)
        print("Output full image size: \'%s\'..." % (str(in_fullimage_array.shape)))

        # 1 step: crop image
        out_cropimage_array = CropImages.compute3D(in_fullimage_array, bounding_box)
        # 2 step: invert image
        out_cropimage_array = FlippingImages.compute(out_cropimage_array, axis=0)
        print("Input cropped image size: \'%s\'..." % (str(out_cropimage_array.shape)))

        out_image_file = joinpathnames(OutputImagesPath, nameOutputImagesFiles(in_image_file))
        print("Output: \'%s\', of dims \'%s\'..." %(basename(out_image_file), str(out_cropimage_array.shape)))

        FileReader.writeImageArray(out_image_file, out_cropimage_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=DATADIR)
    parser.add_argument('inputdir')
    parser.add_argument('outputdir')
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