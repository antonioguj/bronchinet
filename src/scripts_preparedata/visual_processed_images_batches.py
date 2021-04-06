
import numpy as np
import argparse

from common.constant import DATADIR, SIZE_IN_IMAGES, NAME_PROC_IMAGES_RELPATH, NAME_PROC_LABELS_RELPATH, \
    USE_SLIDING_WINDOW_IMAGES, PROP_OVERLAP_SLIDING_WINDOW, USE_RANDOM_WINDOW_IMAGES, NUM_RANDOM_PATCHES_EPOCH, \
    USE_TRANSFORM_RIGID_IMAGES, USE_TRANSFORM_ELASTIC_IMAGES, NAME_REFERENCE_KEYS_PROCIMAGE_FILE
from common.functionutil import join_path_names, basename, list_files_dir, str2bool, str2int, str2tuple_int, \
    str2tuple_float
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

    if (len(list_input_images_files) != len(list_input_labels_files)):
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
                                                        size_in_images=args.size_in_images,
                                                        use_sliding_window_images=args.use_sliding_window_images,
                                                        prop_overlap_slide_window=args.prop_overlap_sliding_window,
                                                        use_transform_rigid_images=args.use_transform_rigid_images,
                                                        use_transform_elastic_images=args.use_transform_elastic_images,
                                                        use_random_window_images=args.use_random_window_images,
                                                        num_random_patches_epoch=args.num_random_patches_epoch,
                                                        batch_size=1,
                                                        is_shuffle=False)
        (image_data_batches, label_data_batches) = image_data_loader.get_full_data()

        num_batches = len(image_data_batches)
        print("\nGenerated total image batches: %s..." % (num_batches))

        for ibatch in range(num_batches):
            out_image_batch = np.squeeze(image_data_batches[ibatch], axis=-1)
            out_label_batch = np.squeeze(label_data_batches[ibatch], axis=-1)

            out_image_filename = join_path_names(output_files_path, name_output_images_files % (ifile + 1, ibatch + 1))
            out_label_filename = join_path_names(output_files_path, name_output_labels_files % (ifile + 1, ibatch + 1))
            print("Output: \'%s\', of dims \'%s\'..." % (basename(out_image_filename), out_image_batch.shape))

            ImageFileReader.write_image(out_image_filename, out_image_batch)
            ImageFileReader.write_image(out_label_filename, out_label_batch)
        # endfor
    # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--output_dir', type=str, default='Visual_ImagePatches/')
    parser.add_argument('--size_in_images', type=str2tuple_int, default=SIZE_IN_IMAGES)
    parser.add_argument('--name_input_images_relpath', type=str, default=NAME_PROC_IMAGES_RELPATH)
    parser.add_argument('--name_input_labels_relpath', type=str, default=NAME_PROC_LABELS_RELPATH)
    parser.add_argument('--use_sliding_window_images', type=str2bool, default=USE_SLIDING_WINDOW_IMAGES)
    parser.add_argument('--prop_overlap_sliding_window', type=str2tuple_float, default=PROP_OVERLAP_SLIDING_WINDOW)
    parser.add_argument('--use_random_window_images', type=str2bool, default=USE_RANDOM_WINDOW_IMAGES)
    parser.add_argument('--num_random_patches_epoch', type=str2int, default=NUM_RANDOM_PATCHES_EPOCH)
    parser.add_argument('--use_transform_rigid_images', type=str2bool, default=USE_TRANSFORM_RIGID_IMAGES)
    parser.add_argument('--use_transform_elastic_images', type=str2bool, default=USE_TRANSFORM_ELASTIC_IMAGES)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
