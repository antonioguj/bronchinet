
from common.constant import *
from common.functionutil import *
from common.workdirmanager import GeneralDirManager
from dataloaders.imagefilereader import ImageFileReader
from imageoperators.boundingboxes import BoundingBoxes
from imageoperators.maskoperator import MaskOperator
from collections import OrderedDict
import argparse



def main(args):

    workdir_manager            = GeneralDirManager(args.datadir)
    input_RoImasks_path        = workdir_manager.get_pathdir_exist(args.name_input_RoImasks_relpath)
    input_reference_files_path = workdir_manager.get_pathdir_exist(args.name_input_reference_files_relpath)
    output_bounding_boxes_file = workdir_manager.get_pathfile_update(args.name_output_bounding_boxes_file)
    list_input_RoImasks_files  = list_files_dir(input_RoImasks_path)
    list_input_reference_files = list_files_dir(input_reference_files_path)



    outdict_crop_bounding_boxes = OrderedDict()
    max_size_bounding_box = (0, 0, 0)
    min_size_bounding_box = (1.0e+03, 1.0e+03, 1.0e+03)

    if args.is_two_bounding_box_each_lungs:
        # compute two bounding-boxes separately, for both left and right lungs
        for i, in_roimask_file in enumerate(list_input_RoImasks_files):
            print("\nInput: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask = ImageFileReader.get_image(in_roimask_file)
            print("Dims : \'%s\'..." % (str(in_roimask.shape)))

            # extract masks corresponding to left (=1) and right (=2) lungs
            (in_roimask_left, in_roimask_right)  = MaskOperator.get_list_masks_with_labels_list(in_roimask, [1, 2])

            # left lung bounding-box:
            bounding_box_left = BoundingBoxes.compute_boundbox_contain_mask(in_roimask_left,
                                                                            size_borders_buffer=args.size_buffer_in_borders,
                                                                            is_boundbox_slices=args.is_calc_bounding_box_in_slices)
            size_bounding_box = BoundingBoxes.get_size_boundbox(bounding_box_left)
            print("Bounding-box around ROI: left lung: \'%s\', of size: \'%s\'" % (bounding_box_left, size_bounding_box))

            max_size_bounding_box = BoundingBoxes.get_max_size_boundbox(size_bounding_box, max_size_bounding_box)
            min_size_bounding_box = BoundingBoxes.get_min_size_boundbox(size_bounding_box, min_size_bounding_box)

            # right lung bounding-box:
            bounding_box_right = BoundingBoxes.compute_boundbox_contain_mask(in_roimask_right,
                                                                             size_borders_buffer=args.size_buffer_in_borders,
                                                                             is_boundbox_slices=args.is_calc_bounding_box_in_slices)
            size_bounding_box = BoundingBoxes.get_size_boundbox(bounding_box_right)
            print("Bounding-box around ROI: right lung: \'%s\', of size: \'%s\'" % (bounding_box_right, size_bounding_box))

            in_reference_key = list_input_reference_files[i]
            outdict_crop_bounding_boxes[basename_filenoext(in_reference_key)] = [bounding_box_left, bounding_box_right]
        # endfor
    else:
        # compute bounding-box including for both left and right lungs
        for i, in_roimask_file in enumerate(list_input_RoImasks_files):
            print("\nInput: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask = ImageFileReader.get_image(in_roimask_file)
            print("Dims : \'%s\'..." % (str(in_roimask.shape)))

            in_roimask = MaskOperator.binarise(in_roimask)  # convert masks to binary (0, 1)

            bounding_box = BoundingBoxes.compute_boundbox_contain_mask(in_roimask,
                                                                       size_borders_buffer=args.size_buffer_in_borders,
                                                                       is_boundbox_slices=args.is_calc_bounding_box_in_slices)
            size_bounding_box = BoundingBoxes.get_size_boundbox(bounding_box)
            print("Bounding-box around ROI: lungs: \'%s\', of size: \'%s\'" % (bounding_box, size_bounding_box))

            max_size_bounding_box = BoundingBoxes.get_max_size_boundbox(size_bounding_box, max_size_bounding_box)
            min_size_bounding_box = BoundingBoxes.get_min_size_boundbox(size_bounding_box, min_size_bounding_box)

            in_reference_key = list_input_reference_files[i]
            outdict_crop_bounding_boxes[basename_filenoext(in_reference_key)] = bounding_box
    # endfor

    print("\nMax size bounding-box found: \'%s\'..." % (str(max_size_bounding_box)))
    print("Min size bounding-box found: \'%s\'..." % (str(min_size_bounding_box)))



    print("\nPost-process the Computed Bounding-Boxes:")
    if args.is_same_size_boundbox_all_images:
        print("Compute bounding-boxes of same size for all images...")

        if args.fixed_size_bounding_box:
            fixed_size_bounding_box = args.fixed_size_bounding_box
            print("Fixed size bounding-box from input: \'%s\'..." % (str(fixed_size_bounding_box)))
        else:
            fixed_size_bounding_box = max_size_bounding_box
            print("Fixed size bounding-box as the max. size of computed bounding-boxes: \'%s\'..." % (str(fixed_size_bounding_box)))
    else:
        fixed_size_bounding_box = False


    for i, (in_key_file, in_list_bounding_boxes) in enumerate(outdict_crop_bounding_boxes.items()):
        print("\nInput Key file: \'%s\'..." % (in_key_file))

        in_roimask_file = list_input_RoImasks_files[i]
        in_shape_roimask = ImageFileReader.get_image_size(in_roimask_file)
        print("Assigned to file: \'%s\', of Dims : \'%s\'..." % (basename(in_roimask_file), str(in_shape_roimask)))

        if not args.is_two_bounding_box_each_lungs:
            in_list_bounding_boxes = [in_list_bounding_boxes]


        for j, in_bounding_box in enumerate(in_list_bounding_boxes):
            size_bounding_box = BoundingBoxes.get_size_boundbox(in_bounding_box)
            print("\nInput Bounding-box: \'%s\', of size: \'%s\'" % (in_bounding_box, size_bounding_box))

            if args.is_same_size_boundbox_all_images:
                if args.is_calc_bounding_box_in_slices:
                    size_proc_bounding_box = (size_bounding_box[0],) + fixed_size_bounding_box
                else:
                    size_proc_bounding_box = fixed_size_bounding_box
                print("Resize bounding-box to fixed final size: \'%s\'..." %(str(size_proc_bounding_box)))
            else:
                size_proc_bounding_box = size_bounding_box

            # Check whether the size of bounding-box is smaller than the training image patches
            if not BoundingBoxes.is_image_inside_boundbox(size_proc_bounding_box, args.size_train_images):
                print("Size of bounding-box is smaller than size of training image patches: \'%s\'..." %(str(args.size_train_images)))

                size_proc_bounding_box = BoundingBoxes.get_max_size_boundbox(size_proc_bounding_box, args.size_train_images)
                print("Resize bounding-box to size: \'%s\'..." %(str(size_proc_bounding_box)))


            if size_proc_bounding_box != size_bounding_box:
                # Compute new bounding-box: of 'size_proc_bounding_box', with same center as 'in_bounding_box', and that fits in 'in_roimask_array'
                proc_bounding_box = BoundingBoxes.calc_boundbox_centered_boundbox_fitimg(in_bounding_box,
                                                                                         size_proc_bounding_box,
                                                                                         in_shape_roimask)
                size_proc_bounding_box = BoundingBoxes.get_size_boundbox(proc_bounding_box)
                print("New processed bounding-box: \'%s\', of size: \'%s\'" % (proc_bounding_box, size_proc_bounding_box))

                if args.is_two_bounding_box_each_lungs:
                    outdict_crop_bounding_boxes[in_key_file][j] = proc_bounding_box
                else:
                    outdict_crop_bounding_boxes[in_key_file] = proc_bounding_box
            else:
                print("No changes made to bounding-box...")
        # endfor
    # endfor

    # Save computed bounding-boxes
    save_dictionary(output_bounding_boxes_file, outdict_crop_bounding_boxes)
    save_dictionary_csv(output_bounding_boxes_file.replace('.npy', '.csv'), outdict_crop_bounding_boxes)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--name_input_RoImasks_relpath', type=str, default=NAME_RAW_ROIMASKS_RELPATH)
    parser.add_argument('--name_input_reference_files_relpath', type=str, default=NAME_REFERENCE_FILES_RELPATH)
    parser.add_argument('--name_output_bounding_boxes_file', type=str, default=NAME_CROP_BOUNDINGBOX_FILE)
    parser.add_argument('--is_two_bounding_box_each_lungs', type=str2bool, default=IS_TWO_BOUNDBOXES_EACH_LUNGS)
    parser.add_argument('--size_buffer_in_borders', type=str2tuple_int, default=SIZE_BUFFER_BOUNDBOX_BORDERS)
    parser.add_argument('--size_train_images', type=str2tuple_int, default=SIZE_IN_IMAGES)
    parser.add_argument('--is_same_size_boundbox_all_images', type=str2bool, default=IS_SAME_SIZE_BOUNDBOX_ALL_IMAGES)
    parser.add_argument('--fixed_size_bounding_box', type=str2tuple_int, default=FIXED_SIZE_BOUNDING_BOX)
    parser.add_argument('--is_calc_bounding_box_in_slices', type=str2bool, default=IS_CALC_BOUNDINGBOX_IN_SLICES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)