
import argparse

from common.constant import BASEDIR, IS_MASK_REGION_INTEREST, IS_CROP_IMAGES, IS_RESCALE_IMAGES, IS_BINARY_TRAIN_MASKS,\
                            NAME_TEMPO_POSTERIORS_RELPATH, NAME_POSTERIORS_RELPATH, NAME_RAW_ROIMASKS_RELPATH, \
                            NAME_REFERENCE_FILES_RELPATH, NAME_REFERENCE_KEYS_POSTERIORS_FILE, \
                            NAME_CROP_BOUNDBOXES_FILE, NAME_RESCALE_FACTORS_FILE
from common.functionutil import is_exist_file, join_path_names, basename, basename_filenoext, list_files_dir, \
                                get_pattern_refer_filename, find_file_inlist_same_prefix, str2bool, read_dictionary, \
                                read_dictionary_configparams
from common.exceptionmanager import catch_error_exception, catch_warning_exception
from common.workdirmanager import TrainDirManager
from dataloaders.imagefilereader import ImageFileReader
from imageoperators.boundingboxes import BoundingBoxes
from imageoperators.imageoperator import ExtendImage, CropAndExtendImage, SetPatchInImage, CropImageAndSetPatchInImage
from imageoperators.maskoperator import MaskOperator


def main(args):

    # SETTINGS
    def name_output_posteriors_files(in_name: str):
        return basename_filenoext(in_name) + '_probmap.nii.gz'
    # --------

    workdir_manager = TrainDirManager(args.basedir)
    input_predictions_path = workdir_manager.get_pathdir_exist(args.name_input_predictions_relpath)
    in_reference_files_path = workdir_manager.get_datadir_exist(args.name_input_reference_files_relpath)
    in_reference_keys_file = workdir_manager.get_pathfile_exist(args.name_input_reference_keys_file)
    output_posteriors_path = workdir_manager.get_pathdir_new(args.name_output_posteriors_relpath)

    list_input_predictions_files = list_files_dir(input_predictions_path)
    indict_reference_keys = read_dictionary(in_reference_keys_file)
    pattern_search_input_files = get_pattern_refer_filename(list(indict_reference_keys.values())[0])

    if (args.is_mask_region_interest):
        input_roimasks_path = workdir_manager.get_datadir_exist(args.name_input_roimasks_relpath)
        list_input_roimasks_files = list_files_dir(input_roimasks_path)

    if (args.is_crop_images):
        input_crop_boundboxes_file = workdir_manager.get_datafile_exist(args.name_crop_boundboxes_file)
        indict_crop_boundboxes = read_dictionary(input_crop_boundboxes_file)

    # if (args.is_rescale_images):
    #     input_rescale_factors_file = workdir_manager.get_datafile_exist(args.name_rescale_factors_file)
    #     indict_rescale_factors = read_dictionary(input_rescale_factors_file)

    if (args.is_crop_images):
        first_elem_dict_crop_boundboxes = list(indict_crop_boundboxes.values())[0]
        if type(first_elem_dict_crop_boundboxes) != list:
            # for new developments, store input dict boundary-boxes per raw images as a list
            # but output only one processed image
            for key, value in indict_crop_boundboxes.items():
                indict_crop_boundboxes[key] = [value]
            # endfor

    # *****************************************************

    # needed to manipulate the list iterator inside the 'for loop'
    list_input_predictions_files = iter(list_input_predictions_files)

    for i, in_prediction_file in enumerate(list_input_predictions_files):
        print("\nInput: \'%s\'..." % (basename(in_prediction_file)))

        inout_prediction = ImageFileReader.get_image(in_prediction_file)
        print("Original dims : \'%s\'..." % (str(inout_prediction.shape)))

        in_reference_key = indict_reference_keys[basename_filenoext(in_prediction_file)]
        in_reference_file = join_path_names(in_reference_files_path, in_reference_key)

        print("Assigned to Reference file: \'%s\'..." % (basename(in_reference_file)))
        in_metadata_file = ImageFileReader.get_image_metadata_info(in_reference_file)

        # ******************************

        if (args.is_crop_images):
            print("Prediction data are cropped. Extend prediction to full image size...")
            out_shape_fullimage = ImageFileReader.get_image_size(in_reference_file)

            list_in_crop_boundboxes = indict_crop_boundboxes[basename_filenoext(in_reference_key)]
            num_crop_boundboxes = len(list_in_crop_boundboxes)

            if num_crop_boundboxes > 1:
                print("A total of \'%s\' cropped predictions are assigned to image: \'%s\'..." % (num_crop_boundboxes,
                                                                                                  in_reference_key))
                for icrop, in_crop_boundox in enumerate(list_in_crop_boundboxes):
                    if icrop > 0:
                        in_next_prediction_file = next(list_input_predictions_files)
                        in_next_prediction = ImageFileReader.get_image(in_next_prediction_file)
                        print("Next input: \'%s\', of dims: \'%s\'..." % (basename(in_next_prediction_file),
                                                                          str(in_next_prediction.shape)))

                    size_in_crop_boundbox = BoundingBoxes.get_size_boundbox(in_crop_boundox)
                    if not BoundingBoxes.is_boundbox_inside_image_size(in_crop_boundox,
                                                                       out_shape_fullimage):
                        print("Bounding-box is larger than image size: : \'%s\' > \'%s\'. Combine cropping with "
                              "extending images..." % (str(size_in_crop_boundbox), str(out_shape_fullimage)))

                        (croppartial_boundbox, extendimg_boundbox) = \
                            BoundingBoxes.calc_boundboxes_crop_extend_image_reverse(in_crop_boundox,
                                                                                    out_shape_fullimage)
                        if icrop == 0:
                            print("Extend image to full size \'%s\' with bounding-box \'%s\': \'%s\'..."
                                  % (str(out_shape_fullimage), icrop, str(in_crop_boundox)))
                            inout_prediction = CropAndExtendImage.compute(inout_prediction,
                                                                          croppartial_boundbox,
                                                                          extendimg_boundbox,
                                                                          out_shape_fullimage)
                        else:
                            print("Set image patch to full size \'%s\' with bounding-box \'%s\': \'%s\'..."
                                  % (str(out_shape_fullimage), icrop, str(in_crop_boundox)))
                            CropImageAndSetPatchInImage.compute(in_next_prediction,
                                                                inout_prediction,
                                                                croppartial_boundbox,
                                                                extendimg_boundbox)
                    else:
                        if icrop == 0:
                            print("Extend image to full size \'%s\' with bounding-box \'%s\': \'%s\'..."
                                  % (str(out_shape_fullimage), icrop, str(in_crop_boundox)))
                            inout_prediction = ExtendImage.compute(inout_prediction,
                                                                   in_crop_boundox,
                                                                   out_shape_fullimage)
                        else:
                            print("Set image patch to full size \'%s\' with bounding-box \'%s\': \'%s\'..."
                                  % (str(out_shape_fullimage), icrop, str(in_crop_boundox)))
                            SetPatchInImage.compute(in_next_prediction,
                                                    inout_prediction,
                                                    in_crop_boundox)
                # endfor
            else:
                in_crop_boundox = list_in_crop_boundboxes[0]
                size_in_crop_boundbox = BoundingBoxes.get_size_boundbox(in_crop_boundox)

                if not BoundingBoxes.is_boundbox_inside_image_size(in_crop_boundox, out_shape_fullimage):
                    print("Bounding-box is larger than image size: : \'%s\' > \'%s\'. Combine cropping with "
                          "extending images..." % (str(size_in_crop_boundbox), str(out_shape_fullimage)))

                    (croppartial_boundbox, extendimg_boundbox) = \
                        BoundingBoxes.calc_boundboxes_crop_extend_image_reverse(in_crop_boundox,
                                                                                out_shape_fullimage)
                    inout_prediction = CropAndExtendImage.compute(inout_prediction,
                                                                  croppartial_boundbox,
                                                                  extendimg_boundbox,
                                                                  out_shape_fullimage)
                else:
                    print("Extend input image to full size \'%s\' with bounding-box: \'%s\'..."
                          % (str(out_shape_fullimage), str(in_crop_boundox)))
                    inout_prediction = ExtendImage.compute(inout_prediction,
                                                           in_crop_boundox,
                                                           out_shape_fullimage)

            print("Final dims: %s..." % (str(inout_prediction.shape)))

        # ******************************

        if (args.is_rescale_images):
            message = 'Rescaling at Post-process time not implemented yet'
            catch_warning_exception(message)

        # ******************************

        if (args.is_mask_region_interest):
            print("Reverse mask to RoI (lungs) in predictions...")
            in_roimask_file = find_file_inlist_same_prefix(basename(in_reference_file), list_input_roimasks_files,
                                                           pattern_prefix=pattern_search_input_files)
            print("RoI mask (lungs) file: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask = ImageFileReader.get_image(in_roimask_file)
            inout_prediction = MaskOperator.mask_image(inout_prediction, in_roimask,
                                                       is_image_mask=args.is_binary_predictions)

        # ******************************

        # Output processed predictions
        output_prediction_file = join_path_names(output_posteriors_path,
                                                 name_output_posteriors_files(in_reference_file))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(output_prediction_file), inout_prediction.shape))

        ImageFileReader.write_image(output_prediction_file, inout_prediction, metadata=in_metadata_file)
    # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--in_config_file', type=str, default=None)
    parser.add_argument('--is_mask_region_interest', type=str2bool, default=IS_MASK_REGION_INTEREST)
    parser.add_argument('--is_crop_images', type=str2bool, default=IS_CROP_IMAGES)
    parser.add_argument('--is_rescale_images', type=str2bool, default=IS_RESCALE_IMAGES)
    parser.add_argument('--is_binary_predictions', type=str2bool, default=IS_BINARY_TRAIN_MASKS)
    parser.add_argument('--name_input_predictions_relpath', type=str, default=NAME_TEMPO_POSTERIORS_RELPATH)
    parser.add_argument('--name_output_posteriors_relpath', type=str, default=NAME_POSTERIORS_RELPATH)
    parser.add_argument('--name_input_roimasks_relpath', type=str, default=NAME_RAW_ROIMASKS_RELPATH)
    parser.add_argument('--name_input_reference_files_relpath', type=str, default=NAME_REFERENCE_FILES_RELPATH)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_POSTERIORS_FILE)
    parser.add_argument('--name_crop_boundboxes_file', type=str, default=NAME_CROP_BOUNDBOXES_FILE)
    parser.add_argument('--name_rescale_factors_file', type=str, default=NAME_RESCALE_FACTORS_FILE)
    args = parser.parse_args()

    if args.in_config_file:
        if not is_exist_file(args.in_config_file):
            message = "Config params file not found: \'%s\'..." % (args.in_config_file)
            catch_error_exception(message)
        else:
            input_args_file = read_dictionary_configparams(args.in_config_file)
        print("Set up experiments with parameters from file: \'%s\'" % (args.in_config_file))
        args.basedir = str(input_args_file['basedir'])
        args.is_mask_region_interest = str2bool(input_args_file['is_mask_region_interest'])
        # args.is_crop_images = str2bool(input_args_file['is_crop_images'])
        # args.name_crop_boundboxes_file = str(input_args_file['name_crop_boundboxes_file'])
        # args.is_rescale_images = str2bool(input_args_file['is_rescale_images'])
        # args.name_rescale_factors_file = str(input_args_file['name_rescale_factors_file'])

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
