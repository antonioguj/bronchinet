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


    workDirsManager  = WorkDirsManager(args.datadir)
    InputImagesPath  = workDirsManager.getNameExistPath(args.nameRawImagesRelPath)
    InReferKeysPath  = workDirsManager.getNameExistPath(args.nameReferKeysRelPath)
    OutputImagesPath = workDirsManager.getNameNewPath  (args.nameProcImagesRelPath)

    listInputImagesFiles = findFilesDirAndCheck(InputImagesPath)
    listInReferKeysFiles = findFilesDirAndCheck(InReferKeysPath)

    if (args.isPrepareLabels):
        InputLabelsPath      = workDirsManager.getNameExistPath(args.nameRawLabelsRelPath)
        OutputLabelsPath     = workDirsManager.getNameNewPath  (args.nameProcLabelsRelPath)
        listInputLabelsFiles = findFilesDirAndCheck(InputLabelsPath)

    if (args.masksToRegionInterest):
        InputRoiMasksPath      = workDirsManager.getNameExistPath(args.nameRawRoiMasksRelPath)
        listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath)

    if (args.isInputExtraLabels):
        InputExtraLabelsPath      = workDirsManager.getNameExistPath(args.nameRawExtraLabelsRelPath)
        OutputExtraLabelsPath     = workDirsManager.getNameNewPath  (args.nameProcExtraLabelsRelPath)
        listInputExtraLabelsFiles = findFilesDirAndCheck(InputExtraLabelsPath)

    if (args.rescaleImages):
        in_rescaleFactors_file = joinpathnames(args.datadir, args.nameRescaleFactorFile)
        dict_rescaleFactors    = readDictionary(in_rescaleFactors_file)

    if (args.cropImages):
        in_cropBoundingBoxes_file = joinpathnames(args.datadir, args.nameCropBoundingBoxFile)
        dict_cropBoundingBoxes    = readDictionary(in_cropBoundingBoxes_file)


    # check that number of files is the same in all dirs
    checkNumFilesInDir = [len(listInputImagesFiles), len(listInReferKeysFiles)]
    if (args.isPrepareLabels):
        checkNumFilesInDir.append( len(listInputLabelsFiles) )
    if (args.masksToRegionInterest):
        checkNumFilesInDir.append( len(listInputRoiMasksFiles) )
    if (args.isInputExtraLabels):
        checkNumFilesInDir.append( len(listInputExtraLabelsFiles) )

    for i in range(1, len(checkNumFilesInDir)):
        if (checkNumFilesInDir[i] != checkNumFilesInDir[0]):
            message = 'num images \'%s\' and labels \'%s\' not equal...' %(checkNumFilesInDir[i], checkNumFilesInDir[0])
            CatchErrorException(message)
    #endfor



    dict_referenceKeys = OrderedDict()

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
            in_referkey_file = listInReferKeysFiles[i]
            rescale_factor = dict_rescaleFactors[filenamenoextension(in_referkey_file)]
            print("Rescale image with a factor: \'%s\'..." %(str(rescale_factor)))

            if rescale_factor != (1.0, 1.0, 1.0):
                inout_image_array = RescaleImages.compute3D(inout_image_array, rescale_factor, order=args.orderInterRescale)

                if (args.isPrepareLabels):
                    inout_label_array = RescaleImages.compute3D(inout_label_array, rescale_factor, order=args.orderInterRescale, is_binary_mask=True)

                if (args.masksToRegionInterest):
                    in_roimask_array = RescaleImages.compute3D(in_roimask_array, rescale_factor, order=args.orderInterRescale, is_binary_mask=True)
                    # remove noise due to interpolation
                    thres_remove_noise = 0.1
                    print("Remove noise in ROI mask by thresholding with value: \'%s\'..." % (thres_remove_noise))
                    in_roimask_array = ThresholdImages.compute(in_roimask_array, thres_val=thres_remove_noise)

                if (args.isInputExtraLabels):
                    inout_extralabel_array = RescaleImages.compute3D(inout_extralabel_array, rescale_factor, order=args.orderInterRescale, is_binary_mask=True)
            else:
                print("Rescale factor (\'%s'\). Skip rescaling..." %(str(rescale_factor)))

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
            in_referkey_file = listInReferKeysFiles[i]
            crop_bounding_box = dict_cropBoundingBoxes[filenamenoextension(in_referkey_file)]
            print("Crop image to bounding-box: \'%s\'..." % (str(crop_bounding_box)))

            if not BoundingBoxes.is_bounding_box_contained_in_image_size(crop_bounding_box, inout_image_array.shape):
                print("Cropping bounding-box: \'%s\' is larger than image size: \'%s\'. Combine cropping with extending images..."
                      %(str(crop_bounding_box), str(inout_image_array.shape)))

                new_inout_image_shape = BoundingBoxes.get_size_bounding_box(crop_bounding_box)
                (croppartial_bounding_box, extendimg_bounding_box) = BoundingBoxes.compute_bounding_boxes_crop_extend_image(crop_bounding_box,
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
                inout_image_array = CropImages.compute3D(inout_image_array, crop_bounding_box)

                if (args.isPrepareLabels):
                    inout_label_array = CropImages.compute3D(inout_label_array, crop_bounding_box)

                if (args.isInputExtraLabels):
                    inout_extralabel_array = CropImages.compute3D(inout_extralabel_array, crop_bounding_box)

            print("Final dims: %s..." %(str(inout_image_array.shape)))
        # *******************************************************************************



        out_image_file = joinpathnames(OutputImagesPath, nameTemplateOutputImagesFiles %(i+1))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_image_file), (inout_image_array.shape)))

        FileReader.writeImageArray(out_image_file, inout_image_array)

        # save this image in reference keys
        dict_referenceKeys[basename(out_image_file)] = basename(in_image_file)

        if (args.isPrepareLabels):
            out_label_file = joinpathnames(OutputLabelsPath, nameTemplateOutputLabelsFiles %(i+1))
            print("And: \'%s\', of dims \'%s\'..." % (basename(out_label_file), str(inout_label_array.shape)))

            FileReader.writeImageArray(out_label_file, inout_label_array)

        if (args.isInputExtraLabels):
            out_extralabel_file = joinpathnames(OutputExtraLabelsPath, nameTemplateOutputExtraLabelsFiles %(i+1))
            print("And: \'%s\', of dims \'%s\'..." % (basename(out_extralabel_file), str(inout_extralabel_array.shape)))

            FileReader.writeImageArray(out_extralabel_file, inout_extralabel_array)
    #endfor


    # Save dictionary in file
    nameoutfile = joinpathnames(args.datadir, args.nameReferenceKeysFile)
    saveDictionary(nameoutfile, dict_referenceKeys)
    nameoutfile = joinpathnames(args.datadir, args.nameReferenceKeysFile.replace('.npy','.csv'))
    saveDictionary_csv(nameoutfile, dict_referenceKeys)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--nameRawImagesRelPath', type=str, default=NAME_RAWIMAGES_RELPATH)
    parser.add_argument('--nameRawLabelsRelPath', type=str, default=NAME_RAWLABELS_RELPATH)
    parser.add_argument('--nameRawRoiMasksRelPath', type=str, default=NAME_RAWROIMASKS_RELPATH)
    parser.add_argument('--nameReferKeysRelPath', type=str, default=NAME_REFERKEYS_RELPATH)
    parser.add_argument('--nameRawExtraLabelsRelPath', type=str, default=NAME_RAWEXTRALABELS_RELPATH)
    parser.add_argument('--nameProcImagesRelPath', type=str, default=NAME_PROCIMAGES_RELPATH)
    parser.add_argument('--nameProcLabelsRelPath', type=str, default=NAME_PROCLABELS_RELPATH)
    parser.add_argument('--nameProcExtraLabelsRelPath', type=str, default=NAME_PROCEXTRALABELS_RELPATH)
    parser.add_argument('--nameReferenceKeysFile', type=str, default=NAME_REFERENCEKEYS_FILE)
    parser.add_argument('--isPrepareLabels', type=str2bool, default=True)
    parser.add_argument('--isInputExtraLabels', type=str2bool, default=False)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--isBinaryTrainMasks', type=str, default=ISBINARYTRAINMASKS)
    parser.add_argument('--rescaleImages', type=str2bool, default=RESCALEIMAGES)
    parser.add_argument('--nameRescaleFactorFile', type=str, default=NAME_RESCALEFACTOR_FILE)
    parser.add_argument('--orderInterRescale', type=int, default=ORDERINTERRESCALE)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    parser.add_argument('--nameCropBoundingBoxFile', type=str, default=NAME_CROPBOUNDINGBOX_FILE)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in sorted(vars(args).iteritems()):
        print("\'%s\' = %s" %(key, value))

    main(args)