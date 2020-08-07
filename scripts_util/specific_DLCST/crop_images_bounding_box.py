
from common.functionutil import *
from dataloaders.imagefilereader import ImageFileReader
from imageoperators.imageoperator import CropImage, FlipImage
import argparse



def main(args):
    # ---------- SETTINGS ----------
    name_output_images_files = lambda in_name: basename_file_noext(in_name) + '.nii.gz'
    # ---------- SETTINGS ----------


    list_input_images_files    = list_files_dir(args.inputdir)
    list_input_reference_files = list_files_dir(args.reference_files_dir)
    dict_input_bounding_boxes  = read_dictionary(args.input_boundboxes_file)
    prefix_pattern_input_files = get_prefix_pattern_filename(list_input_reference_files[0])

    makedir(args.outputdir)



    for i, in_image_file in enumerate(list_input_images_files):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))

        in_full_image = ImageFileReader.get_image(in_image_file)
        print("Original dims: \'%s\'..." % (str(in_full_image.shape)))

        if fileextension(in_image_file) == '.nii.gz':
            inout_metadata = ImageFileReader.get_image_metadata_info(in_image_file)
        else:
            inout_metadata = None

        in_reference_file = find_file_inlist_same_prefix(basename(in_image_file), list_input_reference_files,
                                                         prefix_pattern=prefix_pattern_input_files)
        print("Reference file: \'%s\'..." % (basename(in_reference_file)))


        # 1 step: crop image
        in_bounding_box = dict_input_bounding_boxes[basename_file_noext(in_reference_file)]
        out_crop_image  = CropImage._compute3D(in_full_image, in_bounding_box)
        # 2 step: invert image
        out_crop_image  = FlipImage.compute(out_crop_image, axis=0)


        out_image_file = join_path_names(args.outputdir, name_output_images_files(in_image_file))
        print("Output: \'%s\', of dims \'%s\'..." %(basename(out_image_file), str(out_crop_image.shape)))

        ImageFileReader.write_image(out_image_file, out_crop_image, metadata=inout_metadata)
    # endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir', type=str)
    parser.add_argument('outputdir', type=str)
    parser.add_argument('--reference_files_dir', type=str, default='RawImages/')
    parser.add_argument('--input_boundboxes_file', type=str, default='found_boundingBox_croppedCTinFull.npy')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)