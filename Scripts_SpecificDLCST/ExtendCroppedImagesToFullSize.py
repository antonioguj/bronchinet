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
    nameInputImagesRelPath = 'RawReferResults'
    nameInputReferImagesRelPath = 'Images_Full'
    nameOutputImagesRelPath = 'RawReferResults_Orig_Full'

    nameInputImagesFiles = '*.nii.gz'
    nameInputReferImagesFiles = '*.nii.gz'
    # prefixPatternInputFiles = 'vol[0-9][0-9]_*'
    nameBoundingBoxes = 'found_boundingBoxes_original.npy'
    nameOutputImagesFiles = lambda in_name: filenamenoextension(in_name) + '.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(args.basedir)
    BaseDataPath = workDirsManager.getNameBaseDataPath()
    InputImagesPath = workDirsManager.getNameExistPath(BaseDataPath, nameInputImagesRelPath)
    InputReferImagesPath = workDirsManager.getNameExistPath(BaseDataPath, nameInputReferImagesRelPath)
    OutputImagesPath = workDirsManager.getNameNewPath(BaseDataPath, nameOutputImagesRelPath)

    listInputImagesFiles = findFilesDirAndCheck(InputImagesPath, nameInputImagesFiles)
    listInputReferImagesFiles = findFilesDirAndCheck(InputReferImagesPath, nameInputReferImagesFiles)

    dict_bounding_boxes = readDictionary(joinpathnames(BaseDataPath, nameBoundingBoxes))



    for i, in_image_file in enumerate(listInputImagesFiles):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))

        in_referimage_file = findFileWithSamePrefix(basename(in_image_file), listInputReferImagesFiles,
                                                    prefix_pattern='vol[0-9][0-9]_')
        print("Refer image file: \'%s\'..." % (basename(in_referimage_file)))
        bounding_box = dict_bounding_boxes[filenamenoextension(in_referimage_file)]

        cropped_image_array = FileReader.getImageArray(in_image_file)
        print("Input cropped image size: \'%s\'..." %(str(cropped_image_array.shape)))

        # 1 step: invert image
        cropped_image_array = FlippingImages.compute(cropped_image_array, axis=0)
        # 2 step: extend image
        full_image_shape = FileReader.getImageSize(in_referimage_file)
        full_image_array = ExtendImages.compute3D(cropped_image_array, bounding_box, full_image_shape)
        print("Output full image size: \'%s\'..." % (str(full_image_array.shape)))

        out_image_file = joinpathnames(OutputImagesPath, nameOutputImagesFiles(in_image_file))
        print("Output: \'%s\', of dims \'%s\'..." %(basename(out_image_file), str(full_image_array.shape)))

        FileReader.writeImageArray(out_image_file, full_image_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)