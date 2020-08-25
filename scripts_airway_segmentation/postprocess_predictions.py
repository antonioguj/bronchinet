
from common.constant import *
from common.functionutil import *
from common.workdirmanager import TrainDirManager
from dataloaders.imagefilereader import ImageFileReader
from imageoperators.boundingboxes import BoundingBoxes
from imageoperators.imageoperator import ExtendImage, CropAndExtendImage, SetPatchInImage, CropImageAndSetPatchInImage
from imageoperators.maskoperator import MaskOperator
import argparse



def main(args):
    # ---------- SETTINGS ----------
    name_output_posteriors_files = lambda in_name: basename_file_noext(in_name) + '_probmap.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager             = TrainDirManager(args.basedir)
    input_predictions_path      = workDirsManager.get_pathdir_exist(args.name_input_predictions_relpath)
    in_reference_files_path     = workDirsManager.get_datadir_exist(args.name_input_reference_files_relpath)
    in_reference_keys_file      = workDirsManager.get_pathfile_exist(args.name_input_reference_keys_file)
    output_posteriors_path      = workDirsManager.get_pathdir_new(args.name_output_posteriors_relpath)
    list_input_predictions_files= list_files_dir(input_predictions_path)
    indict_reference_keys       = read_dictionary(in_reference_keys_file)
    prefix_pattern_input_files  = get_prefix_pattern_filename(list(indict_reference_keys.values())[0])

    if (args.is_mask_region_interest):
        input_RoImasks_path       = workDirsManager.get_datadir_exist(args.name_input_RoImasks_relpath)
        list_input_RoImasks_files = list_files_dir(input_RoImasks_path)

    if (args.is_crop_images):
        input_crop_bounding_boxes_file= workDirsManager.get_datafile_exist(args.name_crop_bounding_boxes_file)
        indict_crop_bounding_boxes    = read_dictionary(input_crop_bounding_boxes_file)

    if (args.is_rescale_images):
        input_rescale_factors_file = workDirsManager.get_datafile_exist(args.name_rescale_factors_file)
        indict_rescale_factors     = read_dictionary(input_rescale_factors_file)

    if (args.is_crop_images):
        first_elem_dict_crop_bounding_boxes = list(indict_crop_bounding_boxes.values())[0]
        if type(first_elem_dict_crop_bounding_boxes) != list:
            # for new developments, store input dict boundary-boxes per raw images as a list. But output only one processed image
            for key, value in indict_crop_bounding_boxes.items():
                indict_crop_bounding_boxes[key] = [value]
            # endfor


    # needed to manipulate the list iterator inside the 'for loop'
    list_input_predictions_files = iter(list_input_predictions_files)

    for i, in_prediction_file in enumerate(list_input_predictions_files):
        print("\nInput: \'%s\'..." % (basename(in_prediction_file)))

        inout_prediction = ImageFileReader.get_image(in_prediction_file)
        print("Original dims : \'%s\'..." % (str(inout_prediction.shape)))

        in_reference_key = indict_reference_keys[basename_file_noext(in_prediction_file)]
        in_reference_file = join_path_names(in_reference_files_path, in_reference_key)

        print("Assigned to Reference file: \'%s\'..." % (basename(in_reference_file)))
        in_metadata_file = ImageFileReader.get_image_metadata_info(in_reference_file)


        # *******************************************************************************
        if (args.is_crop_images):
            print("Prediction data are cropped. Extend prediction to full image size...")

            out_shape_fullimage = ImageFileReader.get_image_size(in_reference_file)

            list_in_crop_bounding_boxes = indict_crop_bounding_boxes[basename_file_noext(in_reference_key)]
            num_crop_bounding_boxes = len(list_in_crop_bounding_boxes)

            if num_crop_bounding_boxes > 1:
                print("A total of \'%s\' cropped predictions are assigned to image: \'%s\'..." %(num_crop_bounding_boxes, in_reference_key))

                for j, in_crop_bounding_box in enumerate(list_in_crop_bounding_boxes):
                    if j>0:
                        in_next_prediction_file = next(list_input_predictions_files)
                        in_next_prediction = ImageFileReader.get_image(in_next_prediction_file)
                        print("Next Input: \'%s\', of dims: \'%s\'..." % (basename(in_next_prediction_file), str(in_next_prediction.shape)))

                    size_in_crop_bounding_box = BoundingBoxes.get_size_bounding_box(in_crop_bounding_box)
                    if not BoundingBoxes.is_bounding_box_contained_in_image_size(in_crop_bounding_box, out_shape_fullimage):
                        print("Bounding-box is larger than image size: : \'%s\' > \'%s\'. Combine cropping with extending images..."
                              % (str(size_in_crop_bounding_box), str(out_shape_fullimage)))

                        (croppartial_bounding_box, extendimg_bounding_box) = BoundingBoxes.compute_bounding_boxes_crop_extend_image_reverse(in_crop_bounding_box,
                                                                                                                                            out_shape_fullimage)
                        if j==0:
                            print("Extend Image to full size \'%s\' with bounding-box \'%s\': \'%s\'..." % (str(out_shape_fullimage), j, str(in_crop_bounding_box)))
                            inout_prediction = CropAndExtendImage._compute3D(inout_prediction, croppartial_bounding_box,
                                                                             extendimg_bounding_box, out_shape_fullimage)
                        else:
                            print("Set Image patch to full size \'%s\' with bounding-box \'%s\': \'%s\'..." % (str(out_shape_fullimage), j, str(in_crop_bounding_box)))
                            CropImageAndSetPatchInImage._compute3D(in_next_prediction, inout_prediction,
                                                                   croppartial_bounding_box, extendimg_bounding_box)
                    else:
                        if j==0:
                            print("Extend Image to full size \'%s\' with bounding-box \'%s\': \'%s\'..." % (str(out_shape_fullimage), j, str(in_crop_bounding_box)))
                            inout_prediction = ExtendImage._compute3D(inout_prediction, in_crop_bounding_box, out_shape_fullimage)
                        else:
                            print("Set Image patch to full size \'%s\' with bounding-box \'%s\': \'%s\'..." % (str(out_shape_fullimage), j, str(in_crop_bounding_box)))
                            SetPatchInImage._compute3D(in_next_prediction, inout_prediction, in_crop_bounding_box)
                # endfor
            else:
                in_crop_bounding_box = list_in_crop_bounding_boxes[0]
                size_in_crop_bounding_box = BoundingBoxes.get_size_bounding_box(in_crop_bounding_box)

                if not BoundingBoxes.is_bounding_box_contained_in_image_size(in_crop_bounding_box, out_shape_fullimage):
                    print("Bounding-box is larger than image size: : \'%s\' > \'%s\'. Combine cropping with extending images..."
                          %(str(size_in_crop_bounding_box), str(out_shape_fullimage)))

                    (croppartial_bounding_box, extendimg_bounding_box) = BoundingBoxes.compute_bounding_boxes_crop_extend_image_reverse(in_crop_bounding_box,
                                                                                                                                        out_shape_fullimage)
                    inout_prediction = CropAndExtendImage._compute3D(inout_prediction, croppartial_bounding_box,
                                                                     extendimg_bounding_box, out_shape_fullimage)
                else:
                    print("Extend input Image to full size \'%s\' with bounding-box: \'%s\'..." % (str(out_shape_fullimage),
                                                                                                   str(in_crop_bounding_box)))
                    inout_prediction = ExtendImage._compute3D(inout_prediction, in_crop_bounding_box, out_shape_fullimage)

            print("Final dims: %s..." % (str(inout_prediction.shape)))
        # *******************************************************************************


        # *******************************************************************************
        if (args.is_rescale_images):
            message = 'Rescaling at Post-process time not implemented yet'
            catch_warning_exception(message)
        # *******************************************************************************


        # *******************************************************************************
        if (args.is_mask_region_interest):
            print("Reverse Mask to RoI (lungs) in predictions...")
            in_roimask_file = find_file_inlist_same_prefix(basename(in_reference_file), list_input_RoImasks_files,
                                                           prefix_pattern=prefix_pattern_input_files)
            print("RoI mask (lungs) file: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask = ImageFileReader.get_image(in_roimask_file)
            inout_prediction = MaskOperator.mask_exclude_regions_fillzero(inout_prediction, in_roimask)
        # *******************************************************************************


        # Output processed predictions
        output_prediction_file = join_path_names(output_posteriors_path, name_output_posteriors_files(in_reference_file))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(output_prediction_file), inout_prediction.shape))

        ImageFileReader.write_image(output_prediction_file, inout_prediction, metadata=in_metadata_file)
    # endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--in_config_file', type=str, default=None)
    parser.add_argument('--name_input_predictions_relpath', type=str, default=NAME_TEMPO_POSTERIORS_RELPATH)
    parser.add_argument('--name_input_RoImasks_relpath', type=str, default=NAME_RAW_ROIMASKS_RELPATH)
    parser.add_argument('--name_input_reference_files_relpath', type=str, default=NAME_REFERENCE_FILES_RELPATH)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_POSTERIORS_FILE)
    parser.add_argument('--name_output_posteriors_relpath', type=str, default=NAME_POSTERIORS_RELPATH)
    parser.add_argument('--is_mask_region_interest', type=str2bool, default=IS_MASK_REGION_INTEREST)
    parser.add_argument('--is_crop_images', type=str2bool, default=IS_CROP_IMAGES)
    parser.add_argument('--name_crop_bounding_boxes_file', type=str, default=NAME_CROP_BOUNDINGBOX_FILE)
    parser.add_argument('--is_rescale_images', type=str2bool, default=IS_RESCALE_IMAGES)
    parser.add_argument('--name_rescale_factors_file', type=str, default=NAME_RESCALE_FACTOR_FILE)
    args = parser.parse_args()

    if args.in_config_file:
        if not is_exist_file(args.in_config_file):
            message = "Config params file not found: \'%s\'..." % (args.in_config_file)
            catch_error_exception(message)
        else:
            input_args_file = read_dictionary_configparams(args.in_config_file)
        print("Set up experiments with parameters from file: \'%s\'" %(args.in_config_file))
        args.basedir                        = str(input_args_file['basedir'])
        args.is_mask_region_interest        = str2bool(input_args_file['is_mask_region_interest'])
        #args.is_crop_images                 = str2bool(input_args_file['is_crop_images'])
        #args.name_crop_bounding_boxes_file  = str(input_args_file['name_crop_bounding_boxes_file'])
        #args.is_rescale_images              = str2bool(input_args_file['is_rescale_images'])
        #args.name_rescale_factors_file      = str(input_args_file['name_rescale_factors_file'])

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)
