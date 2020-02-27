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
from OperationImages.OperationImages import *
from OperationImages.BoundingBoxes import *
from OperationImages.OperationMasks import *
from collections import OrderedDict
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameTemplateOutputImagesFiles = 'images_proc-%0.2i.nii.gz'
    nameTemplateOutputLabelsFiles = 'labels_proc-%0.2i.nii.gz'
    nameTemplateOutputExtraLabelsFiles = 'cenlines_proc-%0.2i.nii.gz'
    # ---------- SETTINGS ----------



    workDirsManager    = WorkDirsManager(args.datadir)
    InputImagesPath    = workDirsManager.getNameExistPath(args.nameInputImagesRelPath)
    InputReferKeysPath = workDirsManager.getNameExistPath(args.nameInputReferKeysRelPath)
    OutputImagesPath   = workDirsManager.getNameNewPath  (args.nameOutputImagesRelPath)
    OutputReferKeysFile= workDirsManager.getNameNewUpdateFile(args.nameOutputReferKeysFile)

    listInputImagesFiles    = findFilesDirAndCheck(InputImagesPath)
    listInputReferKeysFiles = findFilesDirAndCheck(InputReferKeysPath)

    if (args.isPrepareLabels):
        InputLabelsPath      = workDirsManager.getNameExistPath(args.nameInputLabelsRelPath)
        OutputLabelsPath     = workDirsManager.getNameNewPath  (args.nameOutputLabelsRelPath)
        listInputLabelsFiles = findFilesDirAndCheck(InputLabelsPath)

        if (len(listInputImagesFiles) != len(listInputLabelsFiles)):
            message = 'num Images \'%s\' and Labels \'%s\' not equal...' %(len(listInputImagesFiles), len(listInputLabelsFiles))
            CatchErrorException(message)

    if (args.masksToRegionInterest):
        InputRoiMasksPath      = workDirsManager.getNameExistPath(args.nameInputRoiMasksRelPath)
        listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath)

        if (len(listInputImagesFiles) != len(listInputRoiMasksFiles)):
            message = 'num Images \'%s\' and Roi Masks \'%s\' not equal...' %(len(listInputImagesFiles), len(listInputRoiMasksFiles))
            CatchErrorException(message)

    if (args.isInputExtraLabels):
        InputExtraLabelsPath      = workDirsManager.getNameExistPath(args.nameInputExtraLabelsRelPath)
        OutputExtraLabelsPath     = workDirsManager.getNameNewPath  (args.nameOutputExtraLabelsRelPath)
        listInputExtraLabelsFiles = findFilesDirAndCheck(InputExtraLabelsPath)

        if (len(listInputImagesFiles) != len(listInputExtraLabelsFiles)):
            message = 'num Images \'%s\' and Extra Labels \'%s\' not equal...' %(len(listInputImagesFiles), len(listInputExtraLabelsFiles))
            CatchErrorException(message)

    if (args.rescaleImages):
        InputRescaleFactorsFile = workDirsManager.getNameExistPath(args.nameRescaleFactorsFile)
        in_dictRescaleFactors   = readDictionary(InputRescaleFactorsFile)

    if (args.cropImages):
        InputCropBoundingBoxesFile= workDirsManager.getNameExistPath(args.nameCropBoundingBoxesFile)
        in_dictCropBoundingBoxes  = readDictionary(InputCropBoundingBoxesFile)



    out_dictReferenceKeys = OrderedDict()

    for i, in_image_file in enumerate(listInputImagesFiles):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))

        # *******************************************************************************
        inout_image_array = FileReader.getImageArray(in_image_file)
        print("Original dims : \'%s\'..." % (str(inout_image_array.shape)))

        if (args.isPrepareLabels):
            in_label_file = listInputLabelsFiles[i]
            print("And Labels: \'%s\'..." % (basename(in_label_file)))

            inout_label_array = FileReader.getImageArray(in_label_file)
            if (args.isBinaryTrainMasks):
                print("Convert masks to binary (0, 1)...")
                inout_label_array = OperationBinaryMasks.process_masks(inout_label_array)

            if inout_label_array.shape != inout_image_array.shape:
                message = "Images and Labels files of different size (not compatible). Skip these data..."
                CatchWarningException(message)
                continue

        if (args.masksToRegionInterest):
            in_roimask_file = listInputRoiMasksFiles[i]
            print("And ROI Mask for labels: \'%s\'..." %(basename(in_roimask_file)))

            in_roimask_array = FileReader.getImageArray(in_roimask_file)
            in_roimask_array = OperationBinaryMasks.process_masks(in_roimask_array)

            if in_roimask_array.shape != inout_image_array.shape:
                message = "Images and ROI Mask files of different size (not compatible). Skip these data..."
                CatchWarningException(message)
                continue

        if (args.isInputExtraLabels):
            in_extralabel_file = listInputExtraLabelsFiles[i]
            print("And extra labels: \'%s\'..." %(basename(in_extralabel_file)))

            inout_extralabel_array = FileReader.getImageArray(in_extralabel_file)
            inout_extralabel_array = OperationBinaryMasks.process_masks(inout_extralabel_array)

            if in_roimask_array.shape != inout_image_array.shape:
                message = "Images and extra Labels files of different size (not compatible). Skip these data..."
                CatchWarningException(message)
                continue
        # *******************************************************************************



        #*******************************************************************************
        if (args.rescaleImages):
            in_referkey_file = listInputReferKeysFiles[i]
            in_rescale_factor = in_dictRescaleFactors[basenameNoextension(in_referkey_file)]
            print("Rescale image with a factor: \'%s\'..." %(str(in_rescale_factor)))

            if in_rescale_factor != (1.0, 1.0, 1.0):
                inout_image_array = RescaleImages.compute3D(inout_image_array, in_rescale_factor, order=args.orderInterRescale)

                if (args.isPrepareLabels):
                    inout_label_array = RescaleImages.compute3D(inout_label_array, in_rescale_factor, order=args.orderInterRescale, is_binary_mask=True)

                if (args.masksToRegionInterest):
                    in_roimask_array = RescaleImages.compute3D(in_roimask_array, in_rescale_factor, order=args.orderInterRescale, is_binary_mask=True)
                    # remove noise due to interpolation
                    thres_remove_noise = 0.1
                    print("Remove noise in ROI mask by thresholding with value: \'%s\'..." % (thres_remove_noise))
                    in_roimask_array = ThresholdImages.compute(in_roimask_array, thres_val=thres_remove_noise)

                if (args.isInputExtraLabels):
                    inout_extralabel_array = RescaleImages.compute3D(inout_extralabel_array, in_rescale_factor, order=args.orderInterRescale, is_binary_mask=True)
            else:
                print("Rescale factor (\'%s'\). Skip rescaling..." %(str(in_rescale_factor)))

            print("Final dims: %s..." %(str(inout_image_array.shape)))
        # *******************************************************************************


        # *******************************************************************************
        if (args.isPrepareLabels and args.masksToRegionInterest):
            print("Mask labels to ROI mask...:")
            inout_label_array = OperationBinaryMasks.apply_mask_exclude_voxels(inout_label_array, in_roimask_array)

            if (args.isInputExtraLabels):
                inout_extralabel_array = OperationBinaryMasks.apply_mask_exclude_voxels(inout_extralabel_array, in_roimask_array)
        # *******************************************************************************


        # *******************************************************************************
        if (args.cropImages):
            in_referkey_file = listInputReferKeysFiles[i]
            in_crop_bounding_box = in_dictCropBoundingBoxes[basenameNoextension(in_referkey_file)]
            print("Crop image to bounding-box: \'%s\'..." % (str(in_crop_bounding_box)))

            if not BoundingBoxes.is_bounding_box_contained_in_image_size(in_crop_bounding_box, inout_image_array.shape):
                print("Cropping bounding-box: \'%s\' is larger than image size: \'%s\'. Combine cropping with extending images..."
                      %(str(in_crop_bounding_box), str(inout_image_array.shape)))

                new_inout_image_shape = BoundingBoxes.get_size_bounding_box(in_crop_bounding_box)
                (croppartial_bounding_box, extendimg_bounding_box) = BoundingBoxes.compute_bounding_boxes_crop_extend_image(in_crop_bounding_box,
                                                                                                                            inout_image_array.shape)
                inout_image_array = CropAndExtendImages.compute3D(inout_image_array, croppartial_bounding_box, extendimg_bounding_box, new_inout_image_shape,
                                                                  background_value=inout_image_array[0][0][0])

                if (args.isPrepareLabels):
                    inout_label_array = CropAndExtendImages.compute3D(inout_label_array, croppartial_bounding_box, extendimg_bounding_box, new_inout_image_shape,
                                                                      background_value=inout_label_array[0][0][0])

                if (args.isInputExtraLabels):
                    inout_extralabel_array = CropAndExtendImages.compute3D(inout_extralabel_array, croppartial_bounding_box, extendimg_bounding_box, new_inout_image_shape,
                                                                           background_value=inout_extralabel_array[0][0][0])
            else:
                inout_image_array = CropImages.compute3D(inout_image_array, in_crop_bounding_box)

                if (args.isPrepareLabels):
                    inout_label_array = CropImages.compute3D(inout_label_array, in_crop_bounding_box)

                if (args.isInputExtraLabels):
                    inout_extralabel_array = CropImages.compute3D(inout_extralabel_array, in_crop_bounding_box)

            print("Final dims: %s..." %(str(inout_image_array.shape)))
        # *******************************************************************************



        output_image_file = joinpathnames(OutputImagesPath, nameTemplateOutputImagesFiles %(i+1))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(output_image_file), (inout_image_array.shape)))

        FileReader.writeImageArray(output_image_file, inout_image_array)

        # save this image in reference keys
        out_dictReferenceKeys[basename(output_image_file)] = basename(in_image_file)

        if (args.isPrepareLabels):
            output_label_file = joinpathnames(OutputLabelsPath, nameTemplateOutputLabelsFiles %(i+1))
            print("And: \'%s\', of dims \'%s\'..." % (basename(output_label_file), str(inout_label_array.shape)))

            FileReader.writeImageArray(output_label_file, inout_label_array)

        if (args.isInputExtraLabels):
            output_extralabel_file = joinpathnames(OutputExtraLabelsPath, nameTemplateOutputExtraLabelsFiles %(i+1))
            print("And: \'%s\', of dims \'%s\'..." % (basename(output_extralabel_file), str(inout_extralabel_array.shape)))

            FileReader.writeImageArray(output_extralabel_file, inout_extralabel_array)
    #endfor


    # Save dictionary in file
    saveDictionary(OutputReferKeysFile, out_dictReferenceKeys)
    saveDictionary_csv(OutputReferKeysFile.replace('.npy','.csv'), out_dictReferenceKeys)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--nameInputImagesRelPath', type=str, default=NAME_RAWIMAGES_RELPATH)
    parser.add_argument('--nameInputLabelsRelPath', type=str, default=NAME_RAWLABELS_RELPATH)
    parser.add_argument('--nameInputRoiMasksRelPath', type=str, default=NAME_RAWROIMASKS_RELPATH)
    parser.add_argument('--nameInputReferKeysRelPath', type=str, default=NAME_REFERKEYS_RELPATH)
    parser.add_argument('--nameInputExtraLabelsRelPath', type=str, default=NAME_RAWEXTRALABELS_RELPATH)
    parser.add_argument('--nameOutputImagesRelPath', type=str, default=NAME_PROCIMAGES_RELPATH)
    parser.add_argument('--nameOutputLabelsRelPath', type=str, default=NAME_PROCLABELS_RELPATH)
    parser.add_argument('--nameOutputExtraLabelsRelPath', type=str, default=NAME_PROCEXTRALABELS_RELPATH)
    parser.add_argument('--nameOutputReferKeysFile', type=str, default=NAME_PROCREFERKEYS_FILE)
    parser.add_argument('--isPrepareLabels', type=str2bool, default=True)
    parser.add_argument('--isInputExtraLabels', type=str2bool, default=False)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--isBinaryTrainMasks', type=str, default=ISBINARYTRAINMASKS)
    parser.add_argument('--rescaleImages', type=str2bool, default=RESCALEIMAGES)
    parser.add_argument('--nameRescaleFactorsFile', type=str, default=NAME_RESCALEFACTOR_FILE)
    parser.add_argument('--orderInterRescale', type=int, default=ORDERINTERRESCALE)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    parser.add_argument('--nameCropBoundingBoxesFile', type=str, default=NAME_CROPBOUNDINGBOX_FILE)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in sorted(vars(args).iteritems()):
        print("\'%s\' = %s" %(key, value))

    main(args)