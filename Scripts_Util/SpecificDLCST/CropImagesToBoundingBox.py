#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from dataloaders.imagefilereader import *
from imageoperators.imageoperator import *
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameOutputImagesFiles = lambda in_name: basename_file_noext(in_name) + '.nii.gz'
    # ---------- SETTINGS ----------


    listInputImagesFiles    = list_files_dir(args.inputdir)
    listInputReferKeysFiles = list_files_dir(args.referkeysdir)
    dictInputBoundingBoxes  = read_dictionary(args.inputBoundBoxesFile)
    prefixPatternInputFiles = get_prefix_pattern_filename(listInputReferKeysFiles[0])

    makedir(args.outputdir)



    for i, in_image_file in enumerate(listInputImagesFiles):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))

        in_fullimage_array = ImageFileReader.get_image(in_image_file)
        print("Original dims: \'%s\'..." % (str(in_fullimage_array.shape)))

        if fileextension(in_image_file) == '.nii.gz':
            inout_metadata = ImageFileReader.get_image_metadata_info(in_image_file)
        else:
            inout_metadata = None

        in_referkey_file = find_file_inlist_same_prefix(basename(in_image_file), listInputReferKeysFiles,
                                                        prefix_pattern=prefixPatternInputFiles)
        print("Reference file: \'%s\'..." % (basename(in_referkey_file)))


        # 1 step: crop image
        in_bounding_box     = dictInputBoundingBoxes[basename_file_noext(in_referkey_file)]
        out_cropimage_array = CropImage._compute3D(in_fullimage_array, in_bounding_box)

        # 2 step: invert image
        out_cropimage_array = FlipImage.compute(out_cropimage_array, axis=0)


        out_image_file = join_path_names(args.outputdir, nameOutputImagesFiles(in_image_file))
        print("Output: \'%s\', of dims \'%s\'..." %(basename(out_image_file), str(out_cropimage_array.shape)))

        ImageFileReader.write_image(out_image_file, out_cropimage_array, metadata=inout_metadata)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str)
    parser.add_argument('outputdir', type=str)
    parser.add_argument('--referkeysdir', type=str, default='RawImages/')
    parser.add_argument('--inputBoundBoxesFile', type=str, default='found_boundingBox_croppedCTinFull.npy')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)