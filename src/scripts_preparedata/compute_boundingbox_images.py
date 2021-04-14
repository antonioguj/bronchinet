
from collections import OrderedDict
import argparse

from common.constant import DATADIR, SIZE_BUFFER_BOUNDBOX_BORDERS, IS_TWO_BOUNDBOXES_LUNGS, SIZE_IN_IMAGES, \
    IS_SAME_SIZE_BOUNDBOX_ALL_IMAGES, SIZE_FIXED_BOUNDBOX_ALL, IS_CALC_BOUNDBOX_IN_SLICES, NAME_RAW_ROIMASKS_RELPATH, \
    NAME_CROP_BOUNDBOXES_FILE, NAME_REFERENCE_FILES_RELPATH
from common.functionutil import basename, basename_filenoext, list_files_dir, str2bool, str2tuple_int, \
    str2tuple_int_none, save_dictionary, save_dictionary_csv
from common.workdirmanager import GeneralDirManager
from dataloaders.imagefilereader import ImageFileReader
from imageoperators.boundingboxes import BoundingBoxes
from imageoperators.maskoperator import MaskOperator


def main(args):

    workdir_manager = GeneralDirManager(args.datadir)
    input_roimasks_path = workdir_manager.get_pathdir_exist(args.name_input_roimasks_relpath)
    input_reference_files_path = workdir_manager.get_pathdir_exist(args.name_input_reference_files_relpath)
    output_boundboxes_file = workdir_manager.get_pathfile_update(args.name_output_boundboxes_file)
    list_input_roimasks_files = list_files_dir(input_roimasks_path)
    list_input_reference_files = list_files_dir(input_reference_files_path)

    # *****************************************************

    outdict_crop_boundboxes = OrderedDict()
    max_size_boundbox = (0, 0, 0)
    min_size_boundbox = (1.0e+03, 1.0e+03, 1.0e+03)

    if args.is_two_boundboxes_lungs:
        # compute two bounding-boxes separately, for both left and right lungs
        for i, in_roimask_file in enumerate(list_input_roimasks_files):
            print("\nInput: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask = ImageFileReader.get_image(in_roimask_file)
            print("Input Dims : \'%s\'..." % (str(in_roimask.shape)))

            # extract masks corresponding to left (=1) and right (=2) lungs
            (in_roimask_left, in_roimask_right) = MaskOperator.get_list_masks_with_labels_list(in_roimask, [1, 2])

            # left lung bounding-box:
            out_boundbox_left = \
                BoundingBoxes.compute_boundbox_contain_mask(in_roimask_left,
                                                            size_borders_buffer=args.size_buffer_in_borders,
                                                            is_boundbox_slices=args.is_calc_boundbox_in_slices)
            size_boundbox = BoundingBoxes.get_size_boundbox(out_boundbox_left)
            print("Bounding-box around ROI: left lung: \'%s\', of size: \'%s\'" % (out_boundbox_left, size_boundbox))

            max_size_boundbox = BoundingBoxes.get_max_size_boundbox(size_boundbox, max_size_boundbox)
            min_size_boundbox = BoundingBoxes.get_min_size_boundbox(size_boundbox, min_size_boundbox)

            # right lung bounding-box:
            out_boundbox_right = \
                BoundingBoxes.compute_boundbox_contain_mask(in_roimask_right,
                                                            size_borders_buffer=args.size_buffer_in_borders,
                                                            is_boundbox_slices=args.is_calc_boundbox_in_slices)
            size_boundbox = BoundingBoxes.get_size_boundbox(out_boundbox_right)
            print("Bounding-box around ROI: right lung: \'%s\', of size: \'%s\'" % (out_boundbox_right, size_boundbox))

            in_reference_key = list_input_reference_files[i]
            outdict_crop_boundboxes[basename_filenoext(in_reference_key)] = [out_boundbox_left, out_boundbox_right]
        # endfor
    else:
        # compute bounding-box including for both left and right lungs
        for i, in_roimask_file in enumerate(list_input_roimasks_files):
            print("\nInput: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask = ImageFileReader.get_image(in_roimask_file)
            print("Input Dims : \'%s\'..." % (str(in_roimask.shape)))

            in_roimask = MaskOperator.binarise(in_roimask)  # convert masks to binary (0, 1)

            out_boundbox = \
                BoundingBoxes.compute_boundbox_contain_mask(in_roimask,
                                                            size_borders_buffer=args.size_buffer_in_borders,
                                                            is_boundbox_slices=args.is_calc_boundbox_in_slices)
            size_boundbox = BoundingBoxes.get_size_boundbox(out_boundbox)
            print("Bounding-box around ROI: lungs: \'%s\', of size: \'%s\'" % (out_boundbox, size_boundbox))

            max_size_boundbox = BoundingBoxes.get_max_size_boundbox(size_boundbox, max_size_boundbox)
            min_size_boundbox = BoundingBoxes.get_min_size_boundbox(size_boundbox, min_size_boundbox)

            in_reference_key = list_input_reference_files[i]
            outdict_crop_boundboxes[basename_filenoext(in_reference_key)] = out_boundbox
    # endfor

    print("\nMax size bounding-box found: \'%s\'..." % (str(max_size_boundbox)))
    print("Min size bounding-box found: \'%s\'..." % (str(min_size_boundbox)))

    # *****************************************************

    print("\nPost-process the Computed Bounding-Boxes:")
    if args.is_same_size_boundbox_all_images:
        print("Compute bounding-boxes of same size for all images...")

        if args.size_fixed_boundbox_all:
            size_fixed_boundbox_all = args.size_fixed_boundbox_all
            print("Fixed size bounding-box from input: \'%s\'..." % (str(size_fixed_boundbox_all)))
        else:
            size_fixed_boundbox_all = max_size_boundbox
            print("Fixed size bounding-box as the max. size of computed bounding-boxes: \'%s\'..."
                  % (str(size_fixed_boundbox_all)))
    else:
        size_fixed_boundbox_all = None

    for i, (in_key_file, in_list_boundboxes) in enumerate(outdict_crop_boundboxes.items()):
        print("\nInput Key file: \'%s\'..." % (in_key_file))

        in_roimask_file = list_input_roimasks_files[i]
        in_shape_roimask = ImageFileReader.get_image_size(in_roimask_file)
        print("Assigned to file: \'%s\', of Dims : \'%s\'..." % (basename(in_roimask_file), str(in_shape_roimask)))

        if not args.is_two_boundboxes_lungs:
            in_list_boundboxes = [in_list_boundboxes]

        for j, in_boundbox in enumerate(in_list_boundboxes):
            size_boundbox = BoundingBoxes.get_size_boundbox(in_boundbox)
            print("\nInput Bounding-box: \'%s\', of size: \'%s\'" % (in_boundbox, size_boundbox))

            if args.is_same_size_boundbox_all_images:
                if args.is_calc_boundbox_in_slices:
                    size_proc_boundbox = (size_boundbox[0],) + size_fixed_boundbox_all
                else:
                    size_proc_boundbox = size_fixed_boundbox_all
                print("Resize bounding-box to fixed final size: \'%s\'..." % (str(size_proc_boundbox)))
            else:
                size_proc_boundbox = size_boundbox

            # Check whether the size of bounding-box is smaller than the training image patches
            if not BoundingBoxes.is_image_inside_boundbox(size_proc_boundbox, args.size_train_images):
                print("Size of bounding-box is smaller than size of training image patches: \'%s\'..."
                      % (str(args.size_train_images)))

                size_proc_boundbox = BoundingBoxes.get_max_size_boundbox(size_proc_boundbox, args.size_train_images)
                print("Resize bounding-box to size: \'%s\'..." % (str(size_proc_boundbox)))

            if size_proc_boundbox != size_boundbox:
                # Compute new bounding-box: of 'size_proc_boundbox', with same center as 'in_boundbox',
                # and that fits in 'in_roimask_array'
                proc_boundbox = BoundingBoxes.calc_boundbox_centered_boundbox_fitimg(in_boundbox,
                                                                                     size_proc_boundbox,
                                                                                     in_shape_roimask)
                size_proc_boundbox = BoundingBoxes.get_size_boundbox(proc_boundbox)
                print("New processed bounding-box: \'%s\', of size: \'%s\'" % (proc_boundbox, size_proc_boundbox))

                if args.is_two_boundboxes_lungs:
                    outdict_crop_boundboxes[in_key_file][j] = proc_boundbox
                else:
                    outdict_crop_boundboxes[in_key_file] = proc_boundbox
            else:
                print("No changes made to bounding-box...")
        # endfor
    # endfor

    # Save computed bounding-boxes
    save_dictionary(output_boundboxes_file, outdict_crop_boundboxes)
    save_dictionary_csv(output_boundboxes_file.replace('.npy', '.csv'), outdict_crop_boundboxes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--size_buffer_in_borders', type=str2tuple_int, default=SIZE_BUFFER_BOUNDBOX_BORDERS)
    parser.add_argument('--is_two_boundboxes_lungs', type=str2bool, default=IS_TWO_BOUNDBOXES_LUNGS)
    parser.add_argument('--size_train_images', type=str2tuple_int, default=SIZE_IN_IMAGES)
    parser.add_argument('--is_same_size_boundbox_all_images', type=str2bool, default=IS_SAME_SIZE_BOUNDBOX_ALL_IMAGES)
    parser.add_argument('--size_fixed_boundbox_all', type=str2tuple_int_none, default=SIZE_FIXED_BOUNDBOX_ALL)
    parser.add_argument('--is_calc_boundbox_in_slices', type=str2bool, default=IS_CALC_BOUNDBOX_IN_SLICES)
    parser.add_argument('--name_input_roimasks_relpath', type=str, default=NAME_RAW_ROIMASKS_RELPATH)
    parser.add_argument('--name_output_boundboxes_file', type=str, default=NAME_CROP_BOUNDBOXES_FILE)
    parser.add_argument('--name_input_reference_files_relpath', type=str, default=NAME_REFERENCE_FILES_RELPATH)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
