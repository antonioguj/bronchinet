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

        in_cropimage_array = FileReader.get_image_array(in_image_file)
        print("Original dims: \'%s\'..." %(str(in_cropimage_array.shape)))

        if fileextension(in_image_file) == '.nii.gz':
            inout_metadata = FileReader.get_image_metadata_info(in_image_file)
        else:
            inout_metadata = None

        in_referkey_file = findFileWithSamePrefixPattern(basename(in_image_file), listInputReferKeysFiles,
                                                         prefix_pattern=prefixPatternInputFiles)
        print("Reference file: \'%s\'..." % (basename(in_referkey_file)))


        # 1 step: invert image
        in_cropimage_array = FlippingImages.compute(in_cropimage_array, axis=0)

        # 2 step: extend image
        in_bounding_box     = dictInputBoundingBoxes[basenameNoextension(in_referkey_file)]
        out_fullimage_shape = FileReader.get_image_size(in_referkey_file)
        out_fullimage_array = ExtendImages.compute3D(in_cropimage_array, in_bounding_box, out_fullimage_shape)


        out_image_file = joinpathnames(args.outputdir, nameOutputImagesFiles(in_image_file))
        print("Output: \'%s\', of dims \'%s\'..." %(basename(out_image_file), str(out_fullimage_array.shape)))

        FileReader.write_image_array(out_image_file, out_fullimage_array, metadata=inout_metadata)
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