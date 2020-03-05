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
from OperationImages.OperationImages import *
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameOutputImagesFiles = lambda in_name: basenameNoextension(in_name) + '.nii.gz'
    # ---------- SETTINGS ----------


    listInputImagesFiles    = findFilesDirAndCheck(args.inputdir)
    listInputReferKeysFiles = findFilesDirAndCheck(args.referkeysdir)
    dictInputBoundingBoxes  = readDictionary(args.inputBoundBoxesFile)
    prefixPatternInputFiles = getFilePrefixPattern(listInputReferKeysFiles[0])

    makedir(args.outputdir)



    for i, in_image_file in enumerate(listInputImagesFiles):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))

        in_fullimage_array = FileReader.getImageArray(in_image_file)
        print("Original dims: \'%s\'..." % (str(in_fullimage_array.shape)))


        in_referkey_file = findFileWithSamePrefixPattern(basename(in_image_file), listInputReferKeysFiles,
                                                         prefix_pattern=prefixPatternInputFiles)
        print("Reference file: \'%s\'..." % (basename(in_referkey_file)))


        # 1 step: crop image
        in_bounding_box     = dictInputBoundingBoxes[basenameNoextension(in_referkey_file)]
        out_cropimage_array = CropImages.compute3D(in_fullimage_array, in_bounding_box)

        # 2 step: invert image
        out_cropimage_array = FlippingImages.compute(out_cropimage_array, axis=0)


        out_image_file = joinpathnames(args.outputdir, nameOutputImagesFiles(in_image_file))
        print("Output: \'%s\', of dims \'%s\'..." %(basename(out_image_file), str(out_cropimage_array.shape)))

        FileReader.writeImageArray(out_image_file, out_cropimage_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str)
    parser.add_argument('outputdir', type=str)
    parser.add_argument('--referkeysdir', type=str, default='RawImages/')
    parser.add_argument('--inputBoundBoxesFile', type=str, default='found_boundingBox_croppedCTinFull.npy')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)