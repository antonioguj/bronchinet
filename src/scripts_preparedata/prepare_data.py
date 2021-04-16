
from typing import List
from collections import OrderedDict
import numpy as np
import argparse

from common.constant import DATADIR, IS_BINARY_TRAIN_MASKS, IS_NORMALIZE_DATA, IS_MASK_REGION_INTEREST, \
    IS_CROP_IMAGES, IS_RESCALE_IMAGES, NAME_RAW_IMAGES_RELPATH, NAME_RAW_LABELS_RELPATH, NAME_PROC_IMAGES_RELPATH, \
    NAME_PROC_LABELS_RELPATH, NAME_RAW_ROIMASKS_RELPATH, NAME_RAW_CENTRELINES_RELPATH, NAME_PROC_EXTRALABELS_RELPATH, \
    NAME_REFERENCE_FILES_RELPATH, NAME_REFERENCE_KEYS_PROCIMAGE_FILE, NAME_CROP_BOUNDBOXES_FILE, \
    NAME_RESCALE_FACTORS_FILE, IS_TWO_BOUNDBOXES_LUNGS
from common.functionutil import join_path_names, basename, basename_filenoext, list_files_dir, str2bool, \
    read_dictionary, save_dictionary, save_dictionary_csv
from common.exceptionmanager import catch_error_exception, catch_warning_exception
from common.workdirmanager import GeneralDirManager
from dataloaders.imagefilereader import ImageFileReader
from imageoperators.imageoperator import CropImage, RescaleImage, CropAndExtendImage, NormaliseImage
from imageoperators.boundingboxes import BoundingBoxes, BoundBox3DType
from imageoperators.maskoperator import MaskOperator


def check_same_number_files_in_list(list_files_1: List[str], list_files_2: List[str]):
    if len(list_files_1) != len(list_files_2):
        message = 'num files in two lists not equal: \'%s\' != \'%s\'...' \
                  % (len(list_files_1), len(list_files_2))
        catch_error_exception(message)


def check_same_size_images(in_image_1: np.ndarray, in_image_2: np.ndarray) -> bool:
    if in_image_1.shape != in_image_2.shape:
        message = "Images have different size: \'%s\' != \'%s\'. Skip these data..." \
                  % (str(in_image_1.shape), str(in_image_2.shape))
        catch_warning_exception(message)
        return True
    else:
        return False


def compute_cropped_patches_list_data(list_in_data: List[np.ndarray],
                                      list_type_in_data: List[str],
                                      in_crop_boundbox: BoundBox3DType,
                                      list_out_data: List[np.ndarray],
                                      list_type_out_data: List[str],
                                      is_insert_new_data: bool = False
                                      ) -> None:
    index_image = list_type_in_data.index('image')
    size_image = list_in_data[index_image].shape
    size_crop_boundbox = BoundingBoxes.get_size_boundbox(in_crop_boundbox)

    if not BoundingBoxes.is_boundbox_inside_image_size(in_crop_boundbox, size_image):
        print("Crop bounding-box is not contained in the size of input images: \'%s\' > \'%s\'. "
              "Extend images after cropping..." % (str(size_crop_boundbox), str(size_image)))
        is_combine_crop_extend = True

        (in_crop_boundbox, in_extend_boundbox) = \
            BoundingBoxes.calc_boundboxes_crop_extend_image(in_crop_boundbox, size_image)
        size_output_image = size_crop_boundbox
    else:
        is_combine_crop_extend = False
        in_extend_boundbox = None
        size_output_image = None

    for idata, (in_data, type_in_data) in enumerate(zip(list_in_data, list_type_in_data)):
        if is_combine_crop_extend:
            print("Crop input data \'%s\' (\'%s\') to bounding-box: \'%s\', and then Extend with bounding-box: "
                  "\'%s\'..." % (idata, type_in_data, str(in_crop_boundbox), str(in_extend_boundbox)))
            out_data = CropAndExtendImage.compute(list_in_data[idata], in_crop_boundbox,
                                                  in_extend_boundbox, size_output_image)
        else:
            print("Crop input data \'%s\' (\'%s\') to bounding-box: \'%s\'..."
                  % (idata, type_in_data, str(in_crop_boundbox)))
            out_data = CropImage.compute(list_in_data[idata], in_crop_boundbox)

        if is_insert_new_data:
            list_out_data.append(out_data)
            list_type_out_data.append(type_in_data)
        else:
            list_out_data[idata] = out_data
    # endfor


def main(args):

    # SETTINGS
    name_template_output_images_files = 'images_proc-%0.2i.nii.gz'
    name_template_output_labels_files = 'labels_proc-%0.2i.nii.gz'
    name_template_output_extra_labels_files = 'cenlines_proc-%0.2i.nii.gz'
    # --------

    workdir_manager = GeneralDirManager(args.datadir)
    input_images_path = workdir_manager.get_pathdir_exist(args.name_input_images_relpath)
    in_reference_files_path = workdir_manager.get_pathdir_exist(args.name_input_reference_files_relpath)
    output_images_path = workdir_manager.get_pathdir_new(args.name_output_images_relpath)
    out_reference_keys_file = workdir_manager.get_pathfile_update(args.name_output_reference_keys_file)
    list_input_images_files = list_files_dir(input_images_path)
    list_in_reference_files = list_files_dir(in_reference_files_path)

    if args.is_prepare_labels:
        input_labels_path = workdir_manager.get_pathdir_exist(args.name_input_labels_relpath)
        output_labels_path = workdir_manager.get_pathdir_new(args.name_output_labels_relpath)
        list_input_labels_files = list_files_dir(input_labels_path)
        check_same_number_files_in_list(list_input_images_files, list_input_labels_files)
    else:
        output_labels_path = None
        list_input_labels_files = None

    if args.is_mask_region_interest:
        input_roimasks_path = workdir_manager.get_pathdir_exist(args.name_input_roimasks_relpath)
        list_input_roimasks_files = list_files_dir(input_roimasks_path)
        check_same_number_files_in_list(list_input_images_files, list_input_roimasks_files)
    else:
        list_input_roimasks_files = None

    if args.is_input_extra_labels:
        input_extra_labels_path = workdir_manager.get_pathdir_exist(args.name_input_extra_labels_relpath)
        output_extra_labels_path = workdir_manager.get_pathdir_new(args.name_output_extra_labels_relpath)
        list_input_extra_labels_files = list_files_dir(input_extra_labels_path)
        check_same_number_files_in_list(list_input_images_files, list_input_extra_labels_files)
    else:
        output_extra_labels_path = None
        list_input_extra_labels_files = None

    if args.is_crop_images:
        input_crop_boundboxes_file = workdir_manager.get_pathfile_exist(args.name_crop_boundboxes_file)
        indict_crop_boundboxes = read_dictionary(input_crop_boundboxes_file)

        if args.is_two_boundboxes_lungs:
            print("Cropping input data to several bounding-boxes. Output several processed images per raw sample...")
            is_output_multiple_files_per_image = True

            def update_template_outfiles(template_name: str) -> str:
                return template_name.replace('.nii.gz', '_crop-%0.2i.nii.gz')
            name_template_output_images_files = update_template_outfiles(name_template_output_images_files)
            name_template_output_labels_files = update_template_outfiles(name_template_output_labels_files)
            name_template_output_extra_labels_files = update_template_outfiles(name_template_output_extra_labels_files)
        else:
            is_output_multiple_files_per_image = False
    else:
        indict_crop_boundboxes = None
        is_output_multiple_files_per_image = False

    is_output_multiple_files_per_label = is_output_multiple_files_per_image

    if args.is_rescale_images:
        input_rescale_factors_file = workdir_manager.get_pathfile_exist(args.name_rescale_factors_file)
        indict_rescale_factors = read_dictionary(input_rescale_factors_file)
    else:
        indict_rescale_factors = None

    # *****************************************************

    # *****************************************************

    outdict_reference_keys = OrderedDict()

    for ifile, in_image_file in enumerate(list_input_images_files):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))

        inout_image = ImageFileReader.get_image(in_image_file)
        print("Input dims : \'%s\'..." % (str(inout_image.shape)))

        list_inout_data = [inout_image]
        list_type_inout_data = ['image']

        # ******************************

        if args.is_prepare_labels:
            in_label_file = list_input_labels_files[ifile]
            print("And Labels: \'%s\'..." % (basename(in_label_file)))

            inout_label = ImageFileReader.get_image(in_label_file)
            if args.is_binary_train_masks:
                print("Convert masks to binary (0, 1)...")
                inout_label = MaskOperator.binarise(inout_label)

            list_inout_data.append(inout_label)
            list_type_inout_data.append('label')

            if check_same_size_images(inout_label, inout_image):
                continue

        # ******************************

        if args.is_mask_region_interest:
            in_roimask_file = list_input_roimasks_files[ifile]
            print("And ROI Mask for labels: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask = ImageFileReader.get_image(in_roimask_file)
            in_roimask = MaskOperator.binarise(in_roimask)
            list_inout_data.append(in_roimask)
            list_type_inout_data.append('roimask')

            if check_same_size_images(in_roimask, inout_image):
                continue

        # ******************************

        if args.is_input_extra_labels:
            in_extra_label_file = list_input_extra_labels_files[ifile]
            print("And extra labels: \'%s\'..." % (basename(in_extra_label_file)))

            inout_extra_label = ImageFileReader.get_image(in_extra_label_file)
            inout_extra_label = MaskOperator.binarise(inout_extra_label)
            list_inout_data.append(inout_extra_label)
            list_type_inout_data.append('extralabel')

            if check_same_size_images(inout_extra_label, inout_image):
                continue

        # ******************************

        if args.is_normalize_data:
            for idata, (in_data, type_in_data) in enumerate(zip(list_inout_data, list_type_inout_data)):
                if type_in_data == 'image' \
                        or (type_in_data == 'label' and not args.is_binary_train_masks):
                    print("Normalize input data \'%s\' (\'%s\')..." % (idata, type_in_data))
                    out_data = NormaliseImage.compute(in_data)
                    list_inout_data[idata] = out_data
            # endfor

        # ******************************

        if args.is_rescale_images:
            in_reference_key = list_in_reference_files[ifile]
            in_rescale_factor = indict_rescale_factors[basename_filenoext(in_reference_key)]
            print("Rescale image with a factor: \'%s\'..." % (str(in_rescale_factor)))

            if in_rescale_factor != (1.0, 1.0, 1.0):
                for idata, (in_data, type_in_data) in enumerate(zip(list_inout_data, list_type_inout_data)):
                    print("Rescale input data \'%s\' (\'%s\')..." % (idata, type_in_data))
                    if type_in_data == 'image':
                        out_data = RescaleImage.compute(in_data, in_rescale_factor, order=3)
                        list_inout_data[idata] = out_data
                    elif type_in_data == 'label':
                        out_data = RescaleImage.compute(in_data, in_rescale_factor, order=3, is_inlabels=True)
                        list_inout_data[idata] = out_data
                    elif type_in_data == 'roimask' \
                            or type_in_data == 'extralabel':
                        out_data = RescaleImage.compute(in_data, in_rescale_factor, order=3, is_inlabels=True,
                                                        is_binarise_output=True)
                        list_inout_data[idata] = out_data
                # endfor
            else:
                print("Rescale factor (\'%s\'). Skip rescaling..." % (str(in_rescale_factor)))

        # ******************************

        if args.is_mask_region_interest:
            index_roimask = list_type_inout_data.index('roimask')

            for idata, (in_data, type_in_data) in enumerate(zip(list_inout_data, list_type_inout_data)):
                if type_in_data == 'label' \
                        or type_in_data == 'extralabel':
                    print("Mask input data \'%s\' (\'%s\') to ROI mask..." % (idata, type_in_data))
                    out_data = MaskOperator.mask_image_exclude_regions(in_data, list_inout_data[index_roimask])
                    list_inout_data[idata] = out_data
            # endfor

            # remove the ROI mask from the list of processing data
            list_inout_data.pop(index_roimask)
            list_type_inout_data.pop(index_roimask)

        # ******************************

        if args.is_crop_images:

            if args.is_two_boundboxes_lungs:
                in_reference_key = list_in_reference_files[ifile]
                inlist_crop_boundboxes = indict_crop_boundboxes[basename_filenoext(in_reference_key)]
                num_crop_boundboxes = len(inlist_crop_boundboxes)
                print("Crop all input data to \'%s\' bounding-boxes: \'%s\' and \'%s\'..."
                      % (num_crop_boundboxes, str(inlist_crop_boundboxes[0]), str(inlist_crop_boundboxes[1])))

                size_inout_data_init = len(list_inout_data)

                for icrop in range(1, num_crop_boundboxes):
                    compute_cropped_patches_list_data(list_inout_data[0:size_inout_data_init],
                                                      list_type_inout_data[0:size_inout_data_init],
                                                      inlist_crop_boundboxes[icrop],
                                                      list_inout_data, list_type_inout_data,
                                                      is_insert_new_data=True)
                # endfor

                # keep the original input image / label until the last cropping, and replace it by the new cropped data
                compute_cropped_patches_list_data(list_inout_data[0:size_inout_data_init],
                                                  list_type_inout_data[0:size_inout_data_init],
                                                  inlist_crop_boundboxes[0],
                                                  list_inout_data, list_type_inout_data,
                                                  is_insert_new_data=False)

                # images / labels created for extra cropping boxes are allocated at the end of 'list_inout_data'
                # push all labels at the end. New order: ('image_0', ..., 'image_N', 'label_0', ..., 'label_N')
                unique_type_data_ordered = ['image', 'label', 'extralabel']
                indexes_reorder_data = []
                for uniq_type in unique_type_data_ordered:
                    indexes_reorder_data += [ind for ind, type in enumerate(list_type_inout_data) if uniq_type in type]

                list_inout_data = [list_inout_data[index] for index in indexes_reorder_data]
                list_type_inout_data = [list_type_inout_data[index] for index in indexes_reorder_data]

            else:
                in_reference_key = list_in_reference_files[ifile]
                in_crop_boundbox = indict_crop_boundboxes[basename_filenoext(in_reference_key)]
                print("Crop all input data to bounding-box: \'%s\'..." % (str(in_crop_boundbox)))

                compute_cropped_patches_list_data(list_inout_data, list_type_inout_data,
                                                  in_crop_boundbox,
                                                  list_inout_data, list_type_inout_data,
                                                  is_insert_new_data=False)

        # ******************************

        # Output processed images
        if args.is_crop_images and args.is_two_boundboxes_lungs:
            first_elem_dict_crop_boundboxes = list(indict_crop_boundboxes.values())[0]
            num_crop_boundboxes_per_image = len(first_elem_dict_crop_boundboxes)
            num_output_files_per_image = num_crop_boundboxes_per_image
        else:
            num_output_files_per_image = 1
        num_output_files_per_label = num_output_files_per_image

        icount = 0
        for isubfile in range(num_output_files_per_image):
            if list_type_inout_data[icount] != 'image':
                message = 'Expected output \'image\', but found type \'%s\'' % (list_type_inout_data[icount])
                catch_error_exception(message)

            if is_output_multiple_files_per_image:
                output_image_file = name_template_output_images_files % (ifile + 1, isubfile + 1)
            else:
                output_image_file = name_template_output_images_files % (ifile + 1)
            output_image_file = join_path_names(output_images_path, output_image_file)

            print("Output \'%s\' image: \'%s\'..." % (icount + 1, basename(output_image_file)))
            ImageFileReader.write_image(output_image_file, list_inout_data[icount])
            icount += 1

            outdict_reference_keys[basename_filenoext(output_image_file)] = basename(in_image_file)
        # endfor

        if args.is_prepare_labels:
            for isubfile in range(num_output_files_per_label):
                if list_type_inout_data[icount] != 'label':
                    message = 'Expected output \'label\', but found type \'%s\'' % (list_type_inout_data[icount])
                    catch_error_exception(message)

                if is_output_multiple_files_per_label:
                    output_label_file = name_template_output_labels_files % (ifile + 1, isubfile + 1)
                else:
                    output_label_file = name_template_output_labels_files % (ifile + 1)
                output_label_file = join_path_names(output_labels_path, output_label_file)

                print("Output \'%s\' label: \'%s\'..." % (icount + 1, basename(output_label_file)))
                ImageFileReader.write_image(output_label_file, list_inout_data[icount])
                icount += 1
            # endfor

        if args.is_input_extra_labels:
            for isubfile in range(num_output_files_per_label):
                if list_type_inout_data[icount] != 'extralabel':
                    message = 'Expected output \'extra_label\', but found type \'%s\'' % (list_type_inout_data[icount])
                    catch_error_exception(message)

                if is_output_multiple_files_per_label:
                    output_label_file = name_template_output_extra_labels_files % (ifile + 1, isubfile + 1)
                else:
                    output_label_file = name_template_output_extra_labels_files % (ifile + 1)
                output_label_file = join_path_names(output_extra_labels_path, output_label_file)

                print("Output \'%s\' extra label: \'%s\'..." % (icount + 1, basename(output_label_file)))
                ImageFileReader.write_image(output_label_file, list_inout_data[icount])
                icount += 1
            # endfor
    # endfor

    # Save reference keys for processed data
    save_dictionary(out_reference_keys_file, outdict_reference_keys)
    save_dictionary_csv(out_reference_keys_file.replace('.npy', '.csv'), outdict_reference_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--is_prepare_labels', type=str2bool, default=True)
    parser.add_argument('--is_input_extra_labels', type=str2bool, default=False)
    parser.add_argument('--is_binary_train_masks', type=str2bool, default=IS_BINARY_TRAIN_MASKS)
    parser.add_argument('--is_normalize_data', type=str2bool, default=IS_NORMALIZE_DATA)
    parser.add_argument('--is_mask_region_interest', type=str2bool, default=IS_MASK_REGION_INTEREST)
    parser.add_argument('--is_crop_images', type=str2bool, default=IS_CROP_IMAGES)
    parser.add_argument('--is_rescale_images', type=str2bool, default=IS_RESCALE_IMAGES)
    parser.add_argument('--is_two_boundboxes_lungs', type=str2bool, default=IS_TWO_BOUNDBOXES_LUNGS)
    parser.add_argument('--name_input_images_relpath', type=str, default=NAME_RAW_IMAGES_RELPATH)
    parser.add_argument('--name_input_labels_relpath', type=str, default=NAME_RAW_LABELS_RELPATH)
    parser.add_argument('--name_output_images_relpath', type=str, default=NAME_PROC_IMAGES_RELPATH)
    parser.add_argument('--name_output_labels_relpath', type=str, default=NAME_PROC_LABELS_RELPATH)
    parser.add_argument('--name_input_roimasks_relpath', type=str, default=NAME_RAW_ROIMASKS_RELPATH)
    parser.add_argument('--name_input_extra_labels_relpath', type=str, default=NAME_RAW_CENTRELINES_RELPATH)
    parser.add_argument('--name_output_extra_labels_relpath', type=str, default=NAME_PROC_EXTRALABELS_RELPATH)
    parser.add_argument('--name_input_reference_files_relpath', type=str, default=NAME_REFERENCE_FILES_RELPATH)
    parser.add_argument('--name_output_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_PROCIMAGE_FILE)
    parser.add_argument('--name_crop_boundboxes_file', type=str, default=NAME_CROP_BOUNDBOXES_FILE)
    parser.add_argument('--name_rescale_factors_file', type=str, default=NAME_RESCALE_FACTORS_FILE)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in sorted(vars(args).items()):
        print("\'%s\' = %s" % (key, value))

    main(args)
