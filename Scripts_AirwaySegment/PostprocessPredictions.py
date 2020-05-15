#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.Constants import *
from Common.WorkDirsManager import *
from DataLoaders.FileReaders import *
from OperationImages.BoundingBoxes import *
from OperationImages.OperationImages import *
from OperationImages.OperationMasks import *
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameOutputPosteriorsFiles = lambda in_name: basenameNoextension(in_name) + '_probmap.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager      = WorkDirsManager(args.basedir)
    InputPredictionsPath = workDirsManager.getNameExistPath        (args.nameInputPredictionsRelPath)
    InputReferKeysPath   = workDirsManager.getNameExistBaseDataPath(args.nameInputReferKeysRelPath)
    InputReferKeysFile   = workDirsManager.getNameExistFile        (args.nameInputReferKeysFile)
    OutputPosteriorsPath = workDirsManager.getNameNewPath          (args.nameOutputPosteriorsRelPath)

    listInputPredictionsFiles = findFilesDirAndCheck(InputPredictionsPath)
    in_dictReferenceKeys      = readDictionary(InputReferKeysFile)
    prefixPatternInputFiles   = getFilePrefixPattern(in_dictReferenceKeys.values()[0])

    if (args.masksToRegionInterest):
        InputRoiMasksPath      = workDirsManager.getNameExistBaseDataPath(args.nameInputRoiMasksRelPath)
        listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath)

    if (args.cropImages):
        InputCropBoundingBoxesFile= workDirsManager.getNameExistBaseDataPath(args.nameCropBoundingBoxesFile)
        in_dictCropBoundingBoxes  = readDictionary(InputCropBoundingBoxesFile)

    if (args.rescaleImages):
        InputRescaleFactorsFile = workDirsManager.getNameExistBaseDataPath(args.nameRescaleFactorsFile)
        in_dictRescaleFactors   = readDictionary(InputRescaleFactorsFile)

    if (args.cropImages):
        first_elem_dictCropBoundingBoxes = in_dictCropBoundingBoxes.values()[0]
        if type(first_elem_dictCropBoundingBoxes) != list:
            # for new developments, store input dict boundary-boxes per raw images as a list. But output only one processed image
            for key, value in in_dictCropBoundingBoxes.iteritems():
                in_dictCropBoundingBoxes[key] = [value]
            #endfor


    # needed to manipulate the list iterator inside the 'for loop'
    listInputPredictionsFiles = iter(listInputPredictionsFiles)

    for i, in_prediction_file in enumerate(listInputPredictionsFiles):
        print("\nInput: \'%s\'..." % (basename(in_prediction_file)))

        inout_prediction_array = FileReader.getImageArray(in_prediction_file)
        print("Original dims : \'%s\'..." % (str(inout_prediction_array.shape)))

        in_referkey_file = in_dictReferenceKeys[basenameNoextension(in_prediction_file)]
        in_reference_file= joinpathnames(InputReferKeysPath, in_referkey_file)

        print("Assigned to Reference file: \'%s\'..." % (basename(in_reference_file)))
        in_metadata_file = FileReader.getImageMetadataInfo(in_reference_file)


        # *******************************************************************************
        if (args.cropImages):
            print("Prediction data are cropped. Extend prediction array to full image size...")

            out_fullimage_shape = FileReader.getImageSize(in_reference_file)

            list_in_crop_bounding_boxes = in_dictCropBoundingBoxes[basenameNoextension(in_referkey_file)]
            num_crop_bounding_boxes = len(list_in_crop_bounding_boxes)

            if num_crop_bounding_boxes > 1:
                print("A total of \'%s\' cropped predictions are assigned to image: \'%s\'..." %(num_crop_bounding_boxes, in_referkey_file))

                for j, in_crop_bounding_box in enumerate(list_in_crop_bounding_boxes):
                    if j>0:
                        in_next_prediction_file = next(listInputPredictionsFiles)
                        in_next_prediction_array = FileReader.getImageArray(in_next_prediction_file)
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
                            inout_prediction_array = CropAndExtendImages.compute3D(inout_prediction_array, croppartial_bounding_box,
                                                                                   extendimg_bounding_box, out_fullimage_shape)
                        else:
                            print("Set array Patch to full size \'%s\' with bounding-box \'%s\': \'%s\'..." % (str(out_fullimage_shape), j, str(in_crop_bounding_box)))
                            CropAndSetPatchInImages.compute3D(in_next_prediction_array, inout_prediction_array,
                                                              croppartial_bounding_box, extendimg_bounding_box)
                    else:
                        if j==0:
                            print("Extend array to full size \'%s\' with bounding-box \'%s\': \'%s\'..." % (str(out_fullimage_shape), j, str(in_crop_bounding_box)))
                            inout_prediction_array = ExtendImages.compute3D(inout_prediction_array, in_crop_bounding_box, out_fullimage_shape)
                        else:
                            print("Set array Patch to full size \'%s\' with bounding-box \'%s\': \'%s\'..." % (str(out_fullimage_shape), j, str(in_crop_bounding_box)))
                            SetPatchInImages.compute3D(in_next_prediction_array, inout_prediction_array, in_crop_bounding_box)
                #endfor
            else:
                in_crop_bounding_box = list_in_crop_bounding_boxes[0]
                size_in_crop_bounding_box = BoundingBoxes.get_size_bounding_box(in_crop_bounding_box)

                if not BoundingBoxes.is_bounding_box_contained_in_image_size(in_crop_bounding_box, out_fullimage_shape):
                    print("Bounding-box is larger than image size: : \'%s\' > \'%s\'. Combine cropping with extending images..."
                          %(str(size_in_crop_bounding_box), str(out_fullimage_shape)))

                    (croppartial_bounding_box, extendimg_bounding_box) = BoundingBoxes.compute_bounding_boxes_crop_extend_image_reverse(in_crop_bounding_box,
                                                                                                                                        out_fullimage_shape)
                    inout_prediction_array = CropAndExtendImages.compute3D(inout_prediction_array, croppartial_bounding_box,
                                                                           extendimg_bounding_box, out_fullimage_shape)
                else:
                    print("Extend input array to full size \'%s\' with bounding-box: \'%s\'..." % (str(out_fullimage_shape),
                                                                                                   str(in_crop_bounding_box)))
                    inout_prediction_array = ExtendImages.compute3D(inout_prediction_array, in_crop_bounding_box, out_fullimage_shape)

            print("Final dims: %s..." % (str(inout_prediction_array.shape)))
        # *******************************************************************************


        # *******************************************************************************
        if (args.rescaleImages):
            message = 'Rescaling at Post-process time not implemented yet'
            CatchWarningException(message)
        # *******************************************************************************


        # *******************************************************************************
        if (args.masksToRegionInterest):
            print("Reverse Mask to RoI (lungs) in predictions...")
            in_roimask_file = findFileWithSamePrefixPattern(basename(in_reference_file), listInputRoiMasksFiles,
                                                            prefix_pattern=prefixPatternInputFiles)
            print("RoI mask (lungs) file: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask_array = FileReader.getImageArray(in_roimask_file)
            inout_prediction_array = OperationBinaryMasks.reverse_mask_exclude_voxels_fillzero(inout_prediction_array, in_roimask_array)
        # *******************************************************************************


        # Output processed predictions
        output_prediction_file = joinpathnames(OutputPosteriorsPath, nameOutputPosteriorsFiles(in_reference_file))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(output_prediction_file), inout_prediction_array.shape))

        FileReader.writeImageArray(output_prediction_file, inout_prediction_array, metadata=in_metadata_file)
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
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    parser.add_argument('--nameCropBoundingBoxesFile', type=str, default=NAME_CROPBOUNDINGBOX_FILE)
    parser.add_argument('--rescaleImages', type=str2bool, default=RESCALEIMAGES)
    parser.add_argument('--nameRescaleFactorsFile', type=str, default=NAME_RESCALEFACTOR_FILE)
    args = parser.parse_args()

    if args.cfgfromfile:
        if not isExistfile(args.cfgfromfile):
            message = "Config params file not found: \'%s\'..." % (args.cfgfromfile)
            CatchErrorException(message)
        else:
            input_args_file = readDictionary_configParams(args.cfgfromfile)
        print("Set up experiments with parameters from file: \'%s\'" %(args.cfgfromfile))
        args.basedir                   = str(input_args_file['basedir'])
        args.masksToRegionInterest     = str2bool(input_args_file['masksToRegionInterest'])
        #args.cropImages                = str2bool(input_args_file['cropImages'])
        #args.nameCropBoundingBoxesFile = str(input_args_file['nameCropBoundingBoxesFile'])
        #args.rescaleImages             = str2bool(input_args_file['rescaleImages'])
        #args.nameRescaleFactorsFile    = str(input_args_file['nameRescaleFactorsFile'])

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
