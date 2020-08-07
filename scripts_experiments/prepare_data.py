
from common.constant import *
from common.functionutil import *
from common.workdirmanager import GeneralDirManager
from dataloaders.imagefilereader import ImageFileReader
from imageoperators.imageoperator import CropImage, RescaleImage, CropAndExtendImage
from imageoperators.boundingboxes import BoundingBoxes
from imageoperators.maskoperator import MaskOperator
from collections import OrderedDict
import argparse


def check_same_number_files_in_list(list_files_1: List[str], list_files_2: List[str]):
    if (len(list_files_1) != len(list_files_2)):
        message = 'num files in two lists not equal: \'%s\' != \'%s\'...' %(len(list_files_1), len(list_files_2))
        catch_error_exception(message)

def check_same_size_images(in_image_1: np.ndarray, in_image_2: np.ndarray) -> bool:
    if in_image_1.shape != in_image_2.shape:
        message = "Images have different size: \'%s\' != \'%s\'. Skip these data..." %(str(in_image_1.shape), str(in_image_2.shape))
        catch_warning_exception(message)
        return True
    else:
        return False



def main(args):
    # ---------- SETTINGS ----------
    name_template_output_images_files = 'images_proc-%0.2i.nii.gz'
    name_template_output_labels_files = 'labels_proc-%0.2i.nii.gz'
    name_template_output_extra_labels_files = 'cenlines_proc-%0.2i.nii.gz'
    # ---------- SETTINGS ----------


    workdir_manager         = GeneralDirManager(args.datadir)
    input_images_path       = workdir_manager.get_pathdir_exist(args.name_input_images_relpath)
    in_reference_files_path = workdir_manager.get_pathdir_exist(args.name_input_reference_files_relpath)
    output_images_path      = workdir_manager.get_pathdir_new(args.name_output_images_relpath)
    out_reference_keys_file = workdir_manager.get_pathfile_update(args.name_output_reference_keys_file)
    list_input_images_files = list_files_dir(input_images_path)
    list_in_reference_files = list_files_dir(in_reference_files_path)

    if (args.is_prepare_labels):
        input_labels_path       = workdir_manager.get_pathdir_exist(args.name_input_labels_relpath)
        output_labels_path      = workdir_manager.get_pathdir_new(args.name_output_labels_relpath)
        list_input_labels_files = list_files_dir(input_labels_path)
        check_same_number_files_in_list(list_input_images_files, list_input_labels_files)

    if (args.is_mask_region_interest):
        input_RoImasks_path       = workdir_manager.get_pathdir_exist(args.name_input_RoImasks_relpath)
        list_input_RoImasks_files = list_files_dir(input_RoImasks_path)
        check_same_number_files_in_list(list_input_images_files, list_input_RoImasks_files)

    if (args.is_input_extra_labels):
        input_extra_labels_path       = workdir_manager.get_pathdir_exist(args.name_input_extra_labels_relpath)
        output_extra_labels_path      = workdir_manager.get_pathdir_new  (args.name_output_extra_labels_relpath)
        list_input_extra_labels_files = list_files_dir(input_extra_labels_path)
        check_same_number_files_in_list(list_input_images_files, list_input_extra_labels_files)

    if (args.is_rescale_images):
        input_rescale_factors_file = workdir_manager.get_pathfile_exist(args.name_rescale_factors_file)
        indict_rescale_factors     = read_dictionary(input_rescale_factors_file)

    if (args.is_crop_images):
        input_crop_bounding_boxes_file = workdir_manager.get_pathfile_exist(args.name_crop_bounding_boxes_file)
        indict_crop_bounding_boxes     = read_dictionary(input_crop_bounding_boxes_file)


    if (args.is_crop_images):
        first_elem_dict_crop_bounding_boxes = list(indict_crop_bounding_boxes.values())[0]
        if type(first_elem_dict_crop_bounding_boxes) == list:
            is_output_multiple_files_per_image = True
            print("\nFound list of crop bounding-boxes per Raw image. Output several processed images...")
            name_template_output_images_files     = 'images_proc-%0.2i_crop-%0.2i.nii.gz'
            name_template_output_labels_files     = 'labels_proc-%0.2i_crop-%0.2i.nii.gz'
            name_template_output_extra_labels_files= 'cenlines_proc-%0.2i_crop-%0.2i.nii.gz'
        else:
            # for new developments, store input dict boundary-boxes per raw images as a list. But output only one processed image
            for key, value in indict_crop_bounding_boxes.items():
                indict_crop_bounding_boxes[key] = [value]
            # endfor
            is_output_multiple_files_per_image = False
    else:
        is_output_multiple_files_per_image = False



    outdict_reference_keys = OrderedDict()

    for i, in_image_file in enumerate(list_input_images_files):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))

        inout_image = ImageFileReader.get_image(in_image_file)
        print("Original dims : \'%s\'..." % (str(inout_image.shape)))

        list_inout_data = [inout_image]
        list_type_inout_data = ['image']


        # *******************************************************************************
        if (args.is_prepare_labels):
            in_label_file = list_input_labels_files[i]
            print("And Labels: \'%s\'..." % (basename(in_label_file)))

            inout_label = ImageFileReader.get_image(in_label_file)
            if (args.is_binary_train_masks):
                print("Convert masks to binary (0, 1)...")
                inout_label = MaskOperator.binarise(inout_label)

            list_inout_data.append(inout_label)
            list_type_inout_data.append('label')

            if check_same_size_images(inout_label, inout_image):
                continue


        if (args.is_mask_region_interest):
            in_roimask_file = list_input_RoImasks_files[i]
            print("And ROI Mask for labels: \'%s\'..." %(basename(in_roimask_file)))

            in_roimask = ImageFileReader.get_image(in_roimask_file)
            if args.is_RoIlabels_multi_RoImasks:
                in_list_roimasks = MaskOperator.get_list_masks_all_labels(in_roimask)
                list_inout_data += in_list_roimasks
                list_type_inout_data += ['roimask'] * len(in_list_roimasks)
            else:
                in_roimask = MaskOperator.binarise(in_roimask)
                list_inout_data.append(in_roimask)
                list_type_inout_data.append('roimask')

            if check_same_size_images(in_roimask, inout_image):
                continue


        if (args.is_input_extra_labels):
            in_extralabel_file = list_input_extra_labels_files[i]
            print("And extra labels: \'%s\'..." %(basename(in_extralabel_file)))

            inout_extralabel = ImageFileReader.get_image(in_extralabel_file)
            inout_extralabel = MaskOperator.binarise(inout_extralabel)
            list_inout_data.append(inout_extralabel)
            list_type_inout_data.append('label')

            if check_same_number_files_in_list(inout_extralabel, inout_image):
                continue

        num_init_labels = list_type_inout_data.count('label')
        # *******************************************************************************


        #*******************************************************************************
        if (args.is_rescale_images):
            in_reference_key = list_in_reference_files[i]
            in_rescale_factor = indict_rescale_factors[basename_file_noext(in_reference_key)]
            print("Rescale image with a factor: \'%s\'..." %(str(in_rescale_factor)))

            if in_rescale_factor != (1.0, 1.0, 1.0):
                for j, (in_data, type_in_data) in enumerate(zip(list_inout_data, list_type_inout_data)):
                    print('Rescale input data \'%s\' of type \'%s\'...' %(j, type_in_data))
                    if type_in_data == 'image':
                        out_data = RescaleImage.compute(in_data, in_rescale_factor, order=3)
                    elif type_in_data == 'label':
                        out_data = RescaleImage.compute(in_data, in_rescale_factor, order=3, is_inlabels=True)
                    elif type_in_data == 'roimask':
                        out_data = RescaleImage.compute(in_data, in_rescale_factor, order=3, is_inlabels=True, is_binarise_output=True)
                    list_inout_data[j] = out_data
                # endfor
            else:
                print("Rescale factor (\'%s'\). Skip rescaling..." %(str(in_rescale_factor)))

            print("Final dims: %s..." %(str(list_inout_data[0].shape)))
        # *******************************************************************************


        # *******************************************************************************
        if (args.is_mask_region_interest):
            list_in_calc_data = [list_inout_data[j] for j, type_data in enumerate(list_type_inout_data) if type_data == 'label']
            list_in_roimasks  = [list_inout_data[j] for j, type_data in enumerate(list_type_inout_data) if type_data == 'roimask']

            list_inout_data = [list_inout_data[0]]
            list_type_inout_data = ['image']

            for j, in_roimask in enumerate(list_in_roimasks):
                for k, in_data in enumerate(list_in_calc_data):
                    print('Masks input labels \'%s\' to ROI masks \'%s\'...' % (k, j))
                    out_data = MaskOperator.mask_exclude_regions(in_data, in_roimask)
                    list_inout_data.append(out_data)
                    list_type_inout_data.append('label')
                # endfor
            # endfor
        # *******************************************************************************


        # *******************************************************************************
        if (args.is_crop_images):
            in_reference_key = list_in_reference_files[i]
            list_in_crop_bounding_boxes = indict_crop_bounding_boxes[basename_file_noext(in_reference_key)]
            num_crop_bounding_boxes = len(list_in_crop_bounding_boxes)
            print("Compute \'%s\' cropped images for this raw image:" %(num_crop_bounding_boxes))

            if (args.is_mask_region_interest and args.is_RoIlabels_multi_RoImasks):
                num_total_labels = list_type_inout_data.count('label')
                num_total_labels_tocrop = num_crop_bounding_boxes * num_init_labels
                if (num_total_labels_tocrop != num_total_labels):
                    message = 'num labels to crop to bounding boxes is wrong: \'%s\' != \'%s\' (expected)...' %(num_total_labels, num_total_labels_tocrop)
                    catch_error_exception(message)

                # In this set-up, there is already computed the input ROI-masked label per cropping bounding-box
                # Insert input image in the right place: before each set of ROI-masked labels
                in_image = list_inout_data[0]
                for j in range(1,num_crop_bounding_boxes):
                    pos_insert = j*(num_init_labels+1)
                    list_inout_data.insert(pos_insert, in_image)
                    list_type_inout_data.insert(pos_insert, 'image')
                # endfor
            else:
                # Concatenate as many input images and labels as num cropping bounding-boxes
                list_inout_data = list_inout_data * num_crop_bounding_boxes
                list_type_inout_data = list_type_inout_data * num_crop_bounding_boxes


            num_images_per_crop_bounding_box = len(list_inout_data) // num_crop_bounding_boxes

            icount = 0
            for j, in_crop_bounding_box in enumerate(list_in_crop_bounding_boxes):
                print("Crop input Images to bounding-box \'%s\' out of total \'%s\': \'%s\'..." % (j, num_crop_bounding_boxes, str(in_crop_bounding_box)))

                size_in_image = list_inout_data[icount].shape
                size_in_crop_bounding_box = BoundingBoxes.get_size_bounding_box(in_crop_bounding_box)

                if not BoundingBoxes.is_bounding_box_contained_in_image_size(in_crop_bounding_box, size_in_image):
                    print("Bounding-box is larger than image size: : \'%s\' > \'%s\'. Combine cropping with extending images..."
                          % (str(size_in_crop_bounding_box), str(size_in_image)))

                    new_size_in_image = size_in_crop_bounding_box
                    (croppartial_bounding_box, extendimg_bounding_box) = BoundingBoxes.compute_bounding_boxes_crop_extend_image(in_crop_bounding_box, size_in_image)

                    for k in range(num_images_per_crop_bounding_box):
                        print('Crop input Image \'%s\' of type \'%s\'...' % (icount, list_type_inout_data[icount]))
                        out_data = CropAndExtendImage._compute3D(list_inout_data[icount], croppartial_bounding_box,
                                                                  extendimg_bounding_box, new_size_in_image)
                        list_inout_data[icount] = out_data
                        icount += 1
                    # endfor
                else:
                    for k in range(num_images_per_crop_bounding_box):
                        print('Crop input Image \'%s\' of type \'%s\'...' % (icount, list_type_inout_data[icount]))
                        out_data = CropImage._compute3D(list_inout_data[icount], in_crop_bounding_box)
                        list_inout_data[icount] = out_data
                        icount += 1
                    # endfor
            # endfor

            print("Final dims: %s..." % (str(list_inout_data[0].shape)))
        # *******************************************************************************


        # Output processed images
        # *******************************************************************************
        if (args.is_crop_images):
            num_output_files_per_image = num_crop_bounding_boxes
        else:
            num_output_files_per_image = 1

        icount = 0
        for j in range(num_output_files_per_image):
            if is_output_multiple_files_per_image:
                output_image_file = join_path_names(output_images_path, name_template_output_images_files % (i + 1, j + 1))
            else:
                output_image_file = join_path_names(output_images_path, name_template_output_images_files % (i + 1))
            print("Output \'%s\' image, of type \'%s\': \'%s\'..." % (icount, list_type_inout_data[icount],
                                                                      basename(output_image_file)))

            ImageFileReader.write_image(output_image_file, list_inout_data[icount])
            icount += 1

            outdict_reference_keys[basename_file_noext(output_image_file)] = basename(in_image_file)

            if (args.is_prepare_labels):
                if is_output_multiple_files_per_image:
                    output_label_file = join_path_names(output_labels_path, name_template_output_labels_files % (i + 1, j + 1))
                else:
                    output_label_file = join_path_names(output_labels_path, name_template_output_labels_files % (i + 1))
                print("Output \'%s\' label, of type \'%s\': \'%s\'..." % (icount, list_type_inout_data[icount],
                                                                          basename(output_label_file)))

                ImageFileReader.write_image(output_label_file, list_inout_data[icount])
                icount += 1

            if (args.is_input_extra_labels):
                if is_output_multiple_files_per_image:
                    output_extralabel_file = join_path_names(output_labels_path, name_template_output_labels_files % (i + 1, j + 1))
                else:
                    output_extralabel_file = join_path_names(output_labels_path, name_template_output_labels_files % (i + 1))
                print("Output \'%s\' extra label, of type \'%s\': \'%s\'..." % (icount, list_type_inout_data[icount],
                                                                                basename(output_extralabel_file)))

                ImageFileReader.write_image(output_extralabel_file, list_inout_data[icount])
                icount += 1
        # endfor
        # *******************************************************************************
    # endfor

    # Save reference keys for processed data
    save_dictionary(out_reference_keys_file, outdict_reference_keys)
    save_dictionary_csv(out_reference_keys_file.replace('.npy', '.csv'), outdict_reference_keys)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--name_input_images_relpath', type=str, default=NAME_RAW_IMAGES_RELPATH)
    parser.add_argument('--name_input_labels_relpath', type=str, default=NAME_RAW_LABELS_RELPATH)
    parser.add_argument('--name_input_RoImasks_relpath', type=str, default=NAME_RAW_ROIMASKS_RELPATH)
    parser.add_argument('--name_input_reference_files_relpath', type=str, default=NAME_REFERENCE_FILES_RELPATH)
    parser.add_argument('--name_input_extra_labels_relpath', type=str, default=NAME_RAW_EXTRALABELS_RELPATH)
    parser.add_argument('--name_output_images_relpath', type=str, default=NAME_PROC_IMAGES_RELPATH)
    parser.add_argument('--name_output_labels_relpath', type=str, default=NAME_PROC_LABELS_RELPATH)
    parser.add_argument('--name_output_extra_labels_relpath', type=str, default=NAME_PROC_EXTRA_LABELS_RELPATH)
    parser.add_argument('--name_output_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    parser.add_argument('--is_prepare_labels', type=str2bool, default=True)
    parser.add_argument('--is_input_extra_labels', type=str2bool, default=False)
    parser.add_argument('--is_binary_train_masks', type=str2bool, default=IS_BINARY_TRAIN_MASKS)
    parser.add_argument('--is_mask_region_interest', type=str2bool, default=IS_MASK_REGION_INTEREST)
    parser.add_argument('--is_rescale_images', type=str2bool, default=IS_RESCALE_IMAGES)
    parser.add_argument('--name_rescale_factors_file', type=str, default=NAME_RESCALE_FACTOR_FILE)
    parser.add_argument('--is_crop_images', type=str2bool, default=IS_CROP_IMAGES)
    parser.add_argument('--name_crop_bounding_boxes_file', type=str, default=NAME_CROP_BOUNDINGBOX_FILE)
    parser.add_argument('--is_RoIlabels_multi_RoImasks', type=str2bool, default=IS_TWO_BOUNDBOXES_EACH_LUNGS)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in sorted(vars(args).items()):
        print("\'%s\' = %s" %(key, value))

    main(args)