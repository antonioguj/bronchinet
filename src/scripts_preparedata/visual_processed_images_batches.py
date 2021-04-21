
import numpy as np
import argparse

from common.constant import DATADIR, SIZE_IN_IMAGES, NAME_PROC_IMAGES_RELPATH, NAME_PROC_LABELS_RELPATH, \
    IS_GENERATE_PATCHES, TYPE_GENERATE_PATCHES, PROP_OVERLAP_SLIDE_WINDOW, NUM_RANDOM_PATCHES_EPOCH, \
    IS_TRANSFORM_IMAGES, TYPE_TRANSFORM_IMAGES, TRANS_RIGID_ROTATION_RANGE, TRANS_RIGID_SHIFT_RANGE, \
    TRANS_RIGID_FLIP_DIRS, TRANS_RIGID_ZOOM_RANGE, TRANS_RIGID_FILL_MODE, NAME_REFERENCE_KEYS_PROCIMAGE_FILE
from common.functionutil import join_path_names, basename, list_files_dir, str2bool, str2int, str2float, \
    str2tuple_bool, str2tuple_int, str2tuple_float
from common.exceptionmanager import catch_error_exception
from common.workdirmanager import GeneralDirManager
from dataloaders.dataloader_manager import get_imagedataloader_2images
from dataloaders.imagefilereader import ImageFileReader


def main(args):

    # SETTINGS
    name_input_images_files = 'images_proc*.nii.gz'
    name_input_labels_files = 'labels_proc*.nii.gz'
    name_output_images_files = 'images_proc-%0.2i_patch-%0.2i.nii.gz'
    name_output_labels_files = 'labels_proc-%0.2i_patch-%0.2i.nii.gz'
    # --------

    workdir_manager = GeneralDirManager(args.datadir)
    input_images_data_path = workdir_manager.get_pathdir_exist(args.name_input_images_relpath)
    input_labels_data_path = workdir_manager.get_pathdir_exist(args.name_input_labels_relpath)
    # in_reference_keys_file = workdir_manager.get_pathfile_exist(args.name_input_reference_keys_file)
    output_files_path = workdir_manager.get_pathdir_new(args.output_dir)
    list_input_images_files = list_files_dir(input_images_data_path, name_input_images_files)
    list_input_labels_files = list_files_dir(input_labels_data_path, name_input_labels_files)

    if len(list_input_images_files) != len(list_input_labels_files):
        message = 'num files in two lists not equal: \'%s\' != \'%s\'...' \
                  % (len(list_input_images_files), len(list_input_labels_files))
        catch_error_exception(message)

    # *****************************************************

    for ifile, (in_image_file, in_label_file) in enumerate(zip(list_input_images_files, list_input_labels_files)):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))
        print("And: \'%s\'..." % (basename(in_label_file)))

        print("Loading data...")
        image_data_loader = get_imagedataloader_2images([in_image_file],
                                                        [in_label_file],
                                                        size_images=args.size_in_images,
                                                        is_generate_patches=args.is_generate_patches,
                                                        type_generate_patches=args.type_generate_patches,
                                                        prop_overlap_slide_images=args.prop_overlap_slide_window,
                                                        num_random_images=args.num_random_patches_epoch,
                                                        is_transform_images=args.is_transform_images,
                                                        type_transform_images=args.type_transform_images,
                                                        trans_rigid_params=args.dict_trans_rigid_parameters,
                                                        batch_size=1,
                                                        is_shuffle=False)
        (image_data_patches, label_data_patches) = image_data_loader.get_full_data()

        num_patches = len(image_data_patches)
        print("\nGenerated total image patches: \'%s\'..." % (num_patches))

        for ipatch in range(num_patches):
            out_image_patch = np.squeeze(image_data_patches[ipatch], axis=-1)
            out_label_patch = np.squeeze(label_data_patches[ipatch], axis=-1)

            out_image_filename = join_path_names(output_files_path, name_output_images_files % (ifile + 1, ipatch + 1))
            out_label_filename = join_path_names(output_files_path, name_output_labels_files % (ifile + 1, ipatch + 1))
            print("Output: \'%s\', of dims \'%s\'..." % (basename(out_image_filename), out_image_patch.shape))

            ImageFileReader.write_image(out_image_filename, out_image_patch)
            ImageFileReader.write_image(out_label_filename, out_label_patch)
        # endfor
    # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--output_dir', type=str, default='Visual_ImagePatches/')
    parser.add_argument('--size_in_images', type=str2tuple_int, default=SIZE_IN_IMAGES)
    parser.add_argument('--name_input_images_relpath', type=str, default=NAME_PROC_IMAGES_RELPATH)
    parser.add_argument('--name_input_labels_relpath', type=str, default=NAME_PROC_LABELS_RELPATH)
    parser.add_argument('--is_generate_patches', type=str2bool, default=IS_GENERATE_PATCHES)
    parser.add_argument('--type_generate_patches', type=str, default=TYPE_GENERATE_PATCHES)
    parser.add_argument('--prop_overlap_slide_window', type=str2tuple_float, default=PROP_OVERLAP_SLIDE_WINDOW)
    parser.add_argument('--num_random_patches_epoch', type=str2int, default=NUM_RANDOM_PATCHES_EPOCH)
    parser.add_argument('--is_transform_images', type=str2bool, default=IS_TRANSFORM_IMAGES)
    parser.add_argument('--type_transform_images', type=str, default=TYPE_TRANSFORM_IMAGES)
    parser.add_argument('--trans_rigid_rotation_range', type=str2tuple_float, default=TRANS_RIGID_ROTATION_RANGE)
    parser.add_argument('--trans_rigid_shift_range', type=str2tuple_float, default=TRANS_RIGID_SHIFT_RANGE)
    parser.add_argument('--trans_rigid_flip_dirs', type=str2tuple_bool, default=TRANS_RIGID_FLIP_DIRS)
    parser.add_argument('--trans_rigid_zoom_range', type=str2float, default=TRANS_RIGID_ZOOM_RANGE)
    parser.add_argument('--trans_rigid_fill_mode', type=str, default=TRANS_RIGID_FILL_MODE)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    args = parser.parse_args()

    args.dict_trans_rigid_parameters = {'rotation_range': args.trans_rigid_rotation_range,
                                        'shift_range': args.trans_rigid_shift_range,
                                        'flip_dirs': args.trans_rigid_flip_dirs,
                                        'zoom_range': args.trans_rigid_zoom_range,
                                        'fill_mode': args.trans_rigid_fill_mode}

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
