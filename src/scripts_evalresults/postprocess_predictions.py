
from typing import Tuple
import numpy as np
import argparse

from common.constant import BASEDIR, IS_MASK_REGION_INTEREST, IS_CROP_IMAGES, IS_RESCALE_IMAGES, IS_BINARY_TRAIN_MASKS,\
    NAME_TEMPO_POSTERIORS_RELPATH, NAME_POSTERIORS_RELPATH, NAME_REFERENCE_FILES_RELPATH, NAME_RAW_ROIMASKS_RELPATH, \
    NAME_REFERENCE_KEYS_POSTERIORS_FILE, NAME_CROP_BOUNDBOXES_FILE, NAME_RESCALE_FACTORS_FILE, IS_TWO_BOUNDBOXES_LUNGS
from common.functionutil import join_path_names, basename, basename_filenoext, list_files_dir, \
    get_regex_pattern_filename, find_file_inlist_with_pattern, str2bool, read_dictionary
from common.exceptionmanager import catch_warning_exception
from common.workdirmanager import TrainDirManager
from dataloaders.imagefilereader import ImageFileReader
from imageoperators.boundingboxes import BoundingBoxes, BoundBox3DType
from imageoperators.imageoperator import ExtendImage, CropAndExtendImage, SetPatchInImage, CropImageAndSetPatchInImage
from imageoperators.maskoperator import MaskOperator


def reconstruct_cropped_patches_first(in_prediction: np.ndarray,
                                      in_crop_boundbox: BoundBox3DType,
                                      size_output_image: Tuple[int, int, int]
                                      ) -> np.ndarray:
    size_crop_boundbox = BoundingBoxes.get_size_boundbox(in_crop_boundbox)

    if not BoundingBoxes.is_boundbox_inside_image_size(in_crop_boundbox, size_output_image):
        print("Crop bounding-box is not contained in the size of output full-size array: \'%s\' > \'%s\'. "
              "Extend images after cropping..." % (str(size_crop_boundbox), str(size_output_image)))

        (in_crop_boundbox, in_extend_boundbox) = \
            BoundingBoxes.calc_boundboxes_crop_extend_image_reverse(in_crop_boundbox, size_output_image)

        print("Crop input prediction to bounding-box: \'%s\', and then extend with bou-box: \'%s\'..."
              % (str(in_crop_boundbox), str(in_extend_boundbox)))
        return CropAndExtendImage.compute(in_prediction, in_crop_boundbox, in_extend_boundbox, size_output_image)

    else:
        print("Extend input prediction with bounding-box: \'%s\'..." % str(in_crop_boundbox))
        return ExtendImage.compute(in_prediction, in_crop_boundbox, size_output_image)


def reconstruct_cropped_patches_next(in_prediction: np.ndarray,
                                     out_fullsize_prediction: np.ndarray,
                                     in_crop_boundbox: BoundBox3DType
                                     ) -> None:
    size_crop_boundbox = BoundingBoxes.get_size_boundbox(in_crop_boundbox)
    size_output_image = out_fullsize_prediction.shape

    if not BoundingBoxes.is_boundbox_inside_image_size(in_crop_boundbox, size_output_image):
        print("Crop bounding-box is not contained in the size of output full-size array: \'%s\' > \'%s\'. "
              "Extend images after cropping..." % (str(size_crop_boundbox), str(size_output_image)))

        (in_crop_boundbox, in_extend_boundbox) = \
            BoundingBoxes.calc_boundboxes_crop_extend_image_reverse(in_crop_boundbox, size_output_image)

        print("Crop input prediction to bounding-box: \'%s\', and then set in full-size array with bou-box: \'%s\'..."
              % (str(in_crop_boundbox), str(in_extend_boundbox)))
        CropImageAndSetPatchInImage.compute(in_prediction, out_fullsize_prediction,
                                            in_crop_boundbox, in_extend_boundbox)

    else:
        print("Set input prediction in full-size array with bounding-box: \'%s\'..." % str(in_crop_boundbox))
        SetPatchInImage.compute(in_prediction, out_fullsize_prediction, in_crop_boundbox)


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
    pattern_search_infiles = get_regex_pattern_filename(list(indict_reference_keys.values())[0])

    if args.is_mask_region_interest:
        input_roimasks_path = workdir_manager.get_datadir_exist(args.name_input_roimasks_relpath)
        list_input_roimasks_files = list_files_dir(input_roimasks_path)
    else:
        list_input_roimasks_files = None

    if args.is_crop_images:
        input_crop_boundboxes_file = workdir_manager.get_datafile_exist(args.name_crop_boundboxes_file)
        indict_crop_boundboxes = read_dictionary(input_crop_boundboxes_file)
    else:
        indict_crop_boundboxes = None

    # if args.is_rescale_images:
    #     input_rescale_factors_file = workdir_manager.get_datafile_exist(args.name_rescale_factors_file)
    #     indict_rescale_factors = read_dictionary(input_rescale_factors_file)
    # else:
    #     indict_rescale_factors = None

    # *****************************************************

    # *****************************************************

    if args.is_two_boundboxes_lungs:
        # to advance the iterator inside the 'for' loop, to process the several cropping patches for the same image
        list_input_predictions_files = iter(list_input_predictions_files)

    for i, in_prediction_file in enumerate(list_input_predictions_files):
        print("\nInput: \'%s\'..." % (basename(in_prediction_file)))

        inout_prediction = ImageFileReader.get_image(in_prediction_file)
        print("Input dims : \'%s\'..." % (str(inout_prediction.shape)))

        in_reference_key = indict_reference_keys[basename_filenoext(in_prediction_file)]
        in_reference_file = join_path_names(in_reference_files_path, in_reference_key)

        print("Assigned to Reference file: \'%s\'..." % (basename(in_reference_file)))
        in_metadata_file = ImageFileReader.get_image_metadata_info(in_reference_file)

        # ******************************

        if args.is_crop_images:
            print("Input data to Network were cropped: Extend prediction to full image size...")

            if args.is_two_boundboxes_lungs:
                inlist_crop_boundboxes = indict_crop_boundboxes[basename_filenoext(in_reference_key)]
                num_crop_boundboxes = len(inlist_crop_boundboxes)
                size_output_fullpred = ImageFileReader.get_image_size(in_reference_file)
                print("Input data cropped to \'%s\' crop bounding-boxes: \'%s\' and \'%s\'..."
                      % (num_crop_boundboxes, str(inlist_crop_boundboxes[0]), str(inlist_crop_boundboxes[1])))

                print("First Patch: Extend cropped prediction to full-size array of size: \'%s\'..."
                      % (str(size_output_fullpred)))
                out_full_prediction = reconstruct_cropped_patches_first(inout_prediction, inlist_crop_boundboxes[0],
                                                                        size_output_fullpred)

                for icrop in range(1, num_crop_boundboxes):
                    # loop over next prediction patches in the list
                    in_prediction_file = next(list_input_predictions_files)
                    inout_prediction = ImageFileReader.get_image(in_prediction_file)
                    print("\nNext Input: \'%s\'..." % (basename(in_prediction_file)))
                    print("Input dims : \'%s\'..." % (str(inout_prediction.shape)))

                    print("Next Patch: Set cropped prediction in the full-size array, with bounding-box: \'%s\'..."
                          % (str(inlist_crop_boundboxes[icrop])))
                    reconstruct_cropped_patches_next(inout_prediction, out_full_prediction,
                                                     inlist_crop_boundboxes[icrop])
                # endfor
                inout_prediction = out_full_prediction

            else:
                in_crop_boundbox = indict_crop_boundboxes[basename_filenoext(in_reference_key)]
                size_output_fullpred = ImageFileReader.get_image_size(in_reference_file)
                print("Input data cropped to bounding-box: \'%s\'..." % (str(in_crop_boundbox)))

                print("First Patch: Extend cropped prediction to full-size array of size: \'%s\'..."
                      % (str(size_output_fullpred)))
                inout_prediction = reconstruct_cropped_patches_first(inout_prediction, in_crop_boundbox,
                                                                     size_output_fullpred)

        # ******************************

        if args.is_rescale_images:
            message = 'Rescaling at Postprocessing time not implemented yet'
            catch_warning_exception(message)

        # ******************************

        if args.is_mask_region_interest:
            print("Input data to Network were masked to ROI (lungs): Reverse mask in predictions...")
            in_roimask_file = find_file_inlist_with_pattern(basename(in_reference_file), list_input_roimasks_files,
                                                            pattern_search=pattern_search_infiles)
            print("ROI mask (lungs) file: \'%s\'..." % (basename(in_roimask_file)))

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
    parser.add_argument('--is_mask_region_interest', type=str2bool, default=IS_MASK_REGION_INTEREST)
    parser.add_argument('--is_crop_images', type=str2bool, default=IS_CROP_IMAGES)
    parser.add_argument('--is_rescale_images', type=str2bool, default=IS_RESCALE_IMAGES)
    parser.add_argument('--is_binary_predictions', type=str2bool, default=IS_BINARY_TRAIN_MASKS)
    parser.add_argument('--is_two_boundboxes_lungs', type=str2bool, default=IS_TWO_BOUNDBOXES_LUNGS)
    parser.add_argument('--name_input_predictions_relpath', type=str, default=NAME_TEMPO_POSTERIORS_RELPATH)
    parser.add_argument('--name_output_posteriors_relpath', type=str, default=NAME_POSTERIORS_RELPATH)
    parser.add_argument('--name_input_reference_files_relpath', type=str, default=NAME_REFERENCE_FILES_RELPATH)
    parser.add_argument('--name_input_reference_keys_file', type=str, default=NAME_REFERENCE_KEYS_POSTERIORS_FILE)
    parser.add_argument('--name_input_roimasks_relpath', type=str, default=NAME_RAW_ROIMASKS_RELPATH)
    parser.add_argument('--name_crop_boundboxes_file', type=str, default=NAME_CROP_BOUNDBOXES_FILE)
    parser.add_argument('--name_rescale_factors_file', type=str, default=NAME_RESCALE_FACTORS_FILE)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
