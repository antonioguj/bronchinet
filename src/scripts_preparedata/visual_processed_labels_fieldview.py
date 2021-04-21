
import numpy as np
import argparse

from common.constant import DATADIR, SIZE_IN_IMAGES, NAME_PROC_IMAGES_RELPATH, NAME_PROC_LABELS_RELPATH, \
    IS_VALID_CONVOLUTIONS, PROP_OVERLAP_SLIDE_WINDOW_TEST, NAME_REFERENCE_KEYS_PROCIMAGE_FILE
from common.functionutil import join_path_names, basename, basename_filenoext, list_files_dir, str2bool, \
    str2tuple_int, str2tuple_float, NetworksUtil
from common.exceptionmanager import catch_warning_exception
from common.workdirmanager import GeneralDirManager
from dataloaders.dataloader_manager import get_imagedataloader_2images
from dataloaders.imagefilereader import ImageFileReader
from models.model_manager import get_network
from postprocessing.postprocessing_manager import get_image_reconstructor


def main(args):

    # SETTINGS
    # name_input_images_files = 'images_proc*.nii.gz'
    name_input_labels_files = 'labels_proc*.nii.gz'

    def name_output_files(in_name, in_size_image):
        suffix = 'outsize-%s' % ('x'.join([str(s) for s in in_size_image]))
        return 'fieldview_' + basename_filenoext(in_name) + '_' + suffix + '.nii.gz'
    # --------

    workdir_manager = GeneralDirManager(args.datadir)
    # input_images_data_path = workdir_manager.get_pathdir_exist(args.name_input_images_relpath)
    input_labels_data_path = workdir_manager.get_pathdir_exist(args.name_input_labels_relpath)
    # in_reference_keys_file = workdir_manager.get_pathfile_exist(args.name_input_reference_keys_file)
    output_files_path = workdir_manager.get_pathdir_new(args.output_dir)
    list_input_labels_files = list_files_dir(input_labels_data_path, name_input_labels_files)

    # Build model to calculate the output size
    model_network = get_network('UNet3DPlugin',
                                args.size_in_images,
                                num_featmaps_in=1,
                                num_channels_in=1,
                                num_classes_out=1,
                                is_use_valid_convols=args.is_valid_convolutions)

    size_output_image_network = model_network.get_size_output_last_layer()

    if args.is_valid_convolutions:
        print("Input size to model: \'%s\'. Output size with Valid Convolutions: \'%s\'..."
              % (str(args.size_in_images), str(size_output_image_network)))

    # Create Image Reconstructor
    image_reconstructor = get_image_reconstructor(args.size_in_images,
                                                  is_reconstruct_patches=True,
                                                  type_reconstruct_patches='slide_window',
                                                  prop_overlap_slide_images=args.prop_overlap_slide_window,
                                                  num_random_images=0,
                                                  is_transform_images=False,
                                                  type_transform_images='',
                                                  trans_rigid_params=None,
                                                  is_nnet_validconvs=args.is_valid_convolutions,
                                                  size_output_images=size_output_image_network,
                                                  is_filter_output_images=args.is_filter_output_network,
                                                  size_filter_output_images=args.size_filter_output_images)

    # *****************************************************

    names_files_different = []

    for ifile, in_label_file in enumerate(list_input_labels_files):
        print("\nInput: \'%s\'..." % (basename(in_label_file)))

        print("Loading data...")
        label_data_loader = get_imagedataloader_2images([in_label_file],
                                                        [in_label_file],
                                                        size_images=args.size_in_images,
                                                        is_generate_patches=True,
                                                        type_generate_patches='slide_window',
                                                        prop_overlap_slide_images=args.prop_overlap_slide_window,
                                                        num_random_images=0,
                                                        is_transform_images=False,
                                                        type_transform_images='',
                                                        trans_rigid_params=None,
                                                        is_nnet_validconvs=args.is_valid_convolutions,
                                                        size_output_images=size_output_image_network,
                                                        batch_size=1,
                                                        is_shuffle=False)
        (_, label_data_patches) = label_data_loader.get_full_data()
        print("Loaded \'%s\' files. Total patches generated: \'%s\'..." % (1, len(label_data_patches)))

        # ******************************

        print("Reconstruct full size Label from sliding-window label patches...")
        shape_reconstructed_label = ImageFileReader.get_image_size(in_label_file)

        image_reconstructor.initialize_recons_data(shape_reconstructed_label)
        image_reconstructor.initialize_recons_array(label_data_patches[0])

        out_label_reconstructed = image_reconstructor.compute_full(label_data_patches)
        out_fieldview_reconstructed = image_reconstructor.get_reconstructed_factor_overlap()

        # ******************************

        print("Compare voxels-wise both Original and Reconstructed Labels...")
        in_label_original = ImageFileReader.get_image(in_label_file)
        in_onlylabels_original = np.where(in_label_original > 0, in_label_original, 0)
        in_onlylabels_reconstructed = np.where(out_label_reconstructed > 0, out_label_reconstructed, 0)

        is_labels_equal_voxelwise = np.array_equal(in_onlylabels_original, in_onlylabels_reconstructed)

        if is_labels_equal_voxelwise:
            print("GOOD: Labels are equal voxelwise...")
        else:
            print("WARNING: Labels 1 and 2 are not equal...")
            num_voxels_labels_original = np.sum(in_onlylabels_original)
            num_voxels_labels_reconstructed = np.sum(in_onlylabels_reconstructed)
            print("Num Voxels for Original and Reconstructed Labels: \'%s\' != \'%s\'..."
                  % (num_voxels_labels_original, num_voxels_labels_reconstructed))
            names_files_different.append(basename(in_label_file))

        # Output computed field of view
        out_filename = join_path_names(output_files_path, name_output_files(in_label_file, size_output_image_network))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_filename), out_fieldview_reconstructed.shape))

        ImageFileReader.write_image(out_filename, out_fieldview_reconstructed)
    # endfor

    if len(names_files_different) == 0:
        print("\nGOOD: ALL IMAGE FILES ARE EQUAL...")
    else:
        print("\nWARNING: Found \'%s\' files that are different. Names of files: \'%s\'..."
              % (len(names_files_different), names_files_different))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--output_dir', type=str, default='Visual_FieldViewNetwork/')
    parser.add_argument('--size_in_images', type=str2tuple_int, default=SIZE_IN_IMAGES)
    parser.add_argument('--name_input_images_relpath', type=str, default=NAME_PROC_IMAGES_RELPATH)
    parser.add_argument('--name_input_labels_relpath', type=str, default=NAME_PROC_LABELS_RELPATH)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    parser.add_argument('--prop_overlap_slide_window', type=str2tuple_float, default=PROP_OVERLAP_SLIDE_WINDOW_TEST)
    parser.add_argument('--is_valid_convolutions', type=str2bool, default=IS_VALID_CONVOLUTIONS)
    parser.add_argument('--is_filter_output_network', type=str2bool, default=False)
    args = parser.parse_args()

    if not args.is_valid_convolutions and not args.is_filter_output_network:
        message = 'Loading network with non-valid convols: better to filter the network output to reduce border effects'
        catch_warning_exception(message)

    if args.is_filter_output_network:
        print("Using the option to filter the network output to reduce border effects...")
        args.size_filter_output_images = NetworksUtil.calc_size_output_layer_valid_convols(args.size_in_images)
        print("Filtering the output images outside the window: %s..." % (str(args.size_filter_output_images)))
    else:
        args.size_filter_output_images = None

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))
    main(args)
