#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from common.constant import *
from common.workdir_manager import *
from dataloaders.imagefilereader import *
from imageoperators.boundingboxes import *
from imageoperators.imageoperator import *
from imageoperators.maskoperator import *
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameOutputPosteriorsFiles = lambda in_name: basename_file_noext(in_name) + '_probmap.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager      = GeneralDirManager(args.basedir)
    InputPredictionsPath = workDirsManager.get_pathdir_exist        (args.nameInputPredictionsRelPath)
    InputReferKeysPath   = workDirsManager.getNameExistBaseDataPath(args.nameInputReferKeysRelPath)
    InputReferKeysFile   = workDirsManager.get_pathfile_exist        (args.nameInputReferKeysFile)
    OutputPosteriorsPath = workDirsManager.get_pathdir_new          (args.nameOutputPosteriorsRelPath)

    listInputPredictionsFiles = list_files_dir(InputPredictionsPath)
    in_dictReferenceKeys      = read_dictionary(InputReferKeysFile)
    prefixPatternInputFiles   = get_prefix_pattern_filename(list(in_dictReferenceKeys.values())[0])


    if (args.masksToRegionInterest):
        InputRoiMasksPath      = workDirsManager.getNameExistBaseDataPath(args.nameInputRoiMasksRelPath)
        listInputRoiMasksFiles = list_files_dir(InputRoiMasksPath)

    if (args.cropImages):
        InputCropBoundingBoxesFile= workDirsManager.getNameExistBaseDataFile(args.nameCropBoundingBoxesFile)
        in_dictCropBoundingBoxes  = read_dictionary(InputCropBoundingBoxesFile)

    if (args.rescaleImages):
        InputRescaleFactorsFile = workDirsManager.getNameExistBaseDataFile(args.nameRescaleFactorsFile)
        in_dictRescaleFactors   = read_dictionary(InputRescaleFactorsFile)

    if (args.cropImages):
        first_elem_dictCropBoundingBoxes = list(in_dictCropBoundingBoxes.values())[0]
        if type(first_elem_dictCropBoundingBoxes) != list:
            # for new developments, store input dict boundary-boxes per raw images as a list. But output only one processed image
            for key, value in in_dictCropBoundingBoxes.items():
                in_dictCropBoundingBoxes[key] = [value]
            #endfor


    # needed to manipulate the list iterator inside the 'for loop'
    listInputPredictionsFiles = iter(listInputPredictionsFiles)

    for i, in_prediction_file in enumerate(listInputPredictionsFiles):
        print("\nInput: \'%s\'..." % (basename(in_prediction_file)))

        inout_prediction_array = ImageFileReader.get_image(in_prediction_file)
        print("Original dims : \'%s\'..." % (str(inout_prediction_array.shape)))

        in_referkey_file = in_dictReferenceKeys[basename_file_noext(in_prediction_file)]
        in_reference_file= join_path_names(InputReferKeysPath, in_referkey_file)

        print("Assigned to Reference file: \'%s\'..." % (basename(in_reference_file)))
        in_metadata_file = ImageFileReader.get_image_metadata_info(in_reference_file)


        # *******************************************************************************
        if (args.cropImages):
            print("Prediction data are cropped. Extend prediction array to full image size...")

            out_fullimage_shape = ImageFileReader.get_image_size(in_reference_file)

            list_in_crop_bounding_boxes = in_dictCropBoundingBoxes[basename_file_noext(in_referkey_file)]
            num_crop_bounding_boxes = len(list_in_crop_bounding_boxes)

            if num_crop_bounding_boxes > 1:
                print("A total of \'%s\' cropped predictions are assigned to image: \'%s\'..." %(num_crop_bounding_boxes, in_referkey_file))

                for j, in_crop_bounding_box in enumerate(list_in_crop_bounding_boxes):
                    if j>0:
                        in_next_prediction_file = next(listInputPredictionsFiles)
                        in_next_prediction_array = ImageFileReader.get_image(in_next_prediction_file)
                        print("Next Input: \'%s\', of dims: \'%s\'..." % (basename(in_next_prediction_file),
                                                                          str(in_next_prediction_array.shape)))

                    size_in_crop_bounding_box = BoundingBoxes.get_size_bounding_box(in_crop_bounding_box)
                    if not BoundingBoxes.is_bounding_box_contained_in_image_size(in_crop_bounding_box, out_fullimage_shape):
                        print("Bounding-box is larger than image size: : \'%s\' > \'%s\'. Combine cropping with extending images..."
                              % (str(size_in_crop_bounding_box), str(out_fullimage_shape)))

                        (croppartial_bounding_box, extendimg_bounding_box) = BoundingBoxes.compute_bounding_boxes_crop_extend_image_reverse(in_crop_bounding_box,
                                                                                                                                            out_fullimage_shape)
                        if j==0:
                            print("Extend array to full size \'%s\' with bounding-box \'%s\': \'%s\'..." % (str(out_fullimage_shape), j, str(in_crop_bounding_box)))
                            inout_prediction_array = CropAndExtendImage._compute3D(inout_prediction_array, croppartial_bounding_box,
                                                                                   extendimg_bounding_box, out_fullimage_shape)
                        else:
                            print("Set array Patch to full size \'%s\' with bounding-box \'%s\': \'%s\'..." % (str(out_fullimage_shape), j, str(in_crop_bounding_box)))
                            CropImageAndSetPatchInImage._compute3D(in_next_prediction_array, inout_prediction_array,
                                                                   croppartial_bounding_box, extendimg_bounding_box)
                    else:
                        if j==0:
                            print("Extend array to full size \'%s\' with bounding-box \'%s\': \'%s\'..." % (str(out_fullimage_shape), j, str(in_crop_bounding_box)))
                            inout_prediction_array = ExtendImage._compute3D(inout_prediction_array, in_crop_bounding_box, out_fullimage_shape)
                        else:
                            print("Set array Patch to full size \'%s\' with bounding-box \'%s\': \'%s\'..." % (str(out_fullimage_shape), j, str(in_crop_bounding_box)))
                            SetPatchInImage._compute3D(in_next_prediction_array, inout_prediction_array, in_crop_bounding_box)
                #endfor
            else:
                in_crop_bounding_box = list_in_crop_bounding_boxes[0]
                size_in_crop_bounding_box = BoundingBoxes.get_size_bounding_box(in_crop_bounding_box)

                if not BoundingBoxes.is_bounding_box_contained_in_image_size(in_crop_bounding_box, out_fullimage_shape):
                    print("Bounding-box is larger than image size: : \'%s\' > \'%s\'. Combine cropping with extending images..."
                          %(str(size_in_crop_bounding_box), str(out_fullimage_shape)))

                    (croppartial_bounding_box, extendimg_bounding_box) = BoundingBoxes.compute_bounding_boxes_crop_extend_image_reverse(in_crop_bounding_box,
                                                                                                                                        out_fullimage_shape)
                    inout_prediction_array = CropAndExtendImage._compute3D(inout_prediction_array, croppartial_bounding_box,
                                                                           extendimg_bounding_box, out_fullimage_shape)
                else:
                    print("Extend input array to full size \'%s\' with bounding-box: \'%s\'..." % (str(out_fullimage_shape),
                                                                                                   str(in_crop_bounding_box)))
                    inout_prediction_array = ExtendImage._compute3D(inout_prediction_array, in_crop_bounding_box, out_fullimage_shape)

            print("Final dims: %s..." % (str(inout_prediction_array.shape)))
        # *******************************************************************************


        # *******************************************************************************
        if (args.rescaleImages):
            message = 'Rescaling at Post-process time not implemented yet'
            catch_warning_exception(message)
        # *******************************************************************************


        # *******************************************************************************
        if (args.masksToRegionInterest):
            print("Reverse Mask to RoI (lungs) in predictions...")
            in_roimask_file = find_file_inlist_same_prefix(basename(in_reference_file), listInputRoiMasksFiles,
                                                           prefix_pattern=prefixPatternInputFiles)
            print("RoI mask (lungs) file: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask_array = ImageFileReader.get_image(in_roimask_file)
            inout_prediction_array = MaskOperator.mask_exclude_regions_fillzero(inout_prediction_array, in_roimask_array)
        # *******************************************************************************


        # Output processed predictions
        output_prediction_file = join_path_names(OutputPosteriorsPath, nameOutputPosteriorsFiles(in_reference_file))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(output_prediction_file), inout_prediction_array.shape))

        ImageFileReader.write_image(output_prediction_file, inout_prediction_array, metadata=in_metadata_file)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--cfgfromfile', type=str, default=None)
    parser.add_argument('--nameInputPredictionsRelPath', type=str, default=NAME_TEMPOPOSTERIORS_RELPATH)
    parser.add_argument('--nameInputRoiMasksRelPath', type=str, default=NAME_RAWROIMASKS_RELPATH)
    parser.add_argument('--nameInputReferKeysRelPath', type=str, default=NAME_REFERKEYS_RELPATH)
    parser.add_argument('--nameInputReferKeysFile', type=str, default=NAME_REFERKEYSPOSTERIORS_FILE)
    parser.add_argument('--nameOutputPosteriorsRelPath', type=str, default=NAME_POSTERIORS_RELPATH)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    parser.add_argument('--nameCropBoundingBoxesFile', type=str, default=NAME_CROPBOUNDINGBOX_FILE)
    parser.add_argument('--rescaleImages', type=str2bool, default=RESCALEIMAGES)
    parser.add_argument('--nameRescaleFactorsFile', type=str, default=NAME_RESCALEFACTOR_FILE)
    args = parser.parse_args()

    if args.cfgfromfile:
        if not is_exist_file(args.cfgfromfile):
            message = "Config params file not found: \'%s\'..." % (args.cfgfromfile)
            catch_error_exception(message)
        else:
            input_args_file = read_dictionary_configparams(args.cfgfromfile)
        print("Set up experiments with parameters from file: \'%s\'" %(args.cfgfromfile))
        args.basedir                   = str(input_args_file['basedir'])
        args.masksToRegionInterest     = str2bool(input_args_file['masksToRegionInterest'])
        #args.cropImages                = str2bool(input_args_file['cropImages'])
        #args.nameCropBoundingBoxesFile = str(input_args_file['nameCropBoundingBoxesFile'])
        #args.rescaleImages             = str2bool(input_args_file['rescaleImages'])
        #args.nameRescaleFactorsFile    = str(input_args_file['nameRescaleFactorsFile'])

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)
