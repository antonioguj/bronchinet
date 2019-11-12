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
from Preprocessing.OperationImages import *
from Preprocessing.OperationMasks import *
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameInputImagesRelPath   = 'RawImages/'
    nameInputLabelsRelPath   = 'RawAirways/'
    nameInputRoiMasksRelPath = 'RawLungs/'
    nameReferenceFilesRelPath= 'RawImages/'
    nameOutputImagesRelPath  = 'Images_WorkData_Rescaled/'
    nameOutputLabelsRelPath  = 'Labels_WorkData_Rescaled/'
    nameOutputImagesFiles    = 'images_proc-%0.2i.nii.gz'
    nameOutputLabelsFiles    = 'labels_proc-%0.2i.nii.gz'

    nameCropBoundingBoxes = 'cropBoundingBoxes_images.npy'
    nameRescaleFactors    = 'rescaleFactors_images.npy'

    if args.isInputExtraLabels:
        nameInputExtraLabelsRelPath  = 'RawCentrelines/'
        nameOutputExtraLabelsRelPath = 'ExtraLabels_WorkData/'
        nameOutputExtraLabelsFiles   = 'cenlines_proc-%0.2i.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager    = WorkDirsManager(args.datadir)
    InputImagesPath    = workDirsManager.getNameExistPath(nameInputImagesRelPath )
    InputLabelsPath    = workDirsManager.getNameExistPath(nameInputLabelsRelPath )
    ReferenceFilesPath = workDirsManager.getNameExistPath(nameReferenceFilesRelPath)
    OutputImagesPath   = workDirsManager.getNameNewPath  (nameOutputImagesRelPath)
    OutputLabelsPath   = workDirsManager.getNameNewPath  (nameOutputLabelsRelPath)

    listInputImagesFiles = findFilesDirAndCheck(InputImagesPath)
    listInputLabelsFiles = findFilesDirAndCheck(InputLabelsPath)
    listReferenceFiles   = findFilesDirAndCheck(ReferenceFilesPath)

    if (args.masksToRegionInterest):
        InputRoiMasksPath      = workDirsManager.getNameExistPath(nameInputRoiMasksRelPath)
        listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath)

    if (args.isInputExtraLabels):
        InputExtraLabelsPath      = workDirsManager.getNameExistPath(nameInputExtraLabelsRelPath)
        listInputExtraLabelsFiles = findFilesDirAndCheck(InputExtraLabelsPath)
        OutputExtraLabelsPath     = workDirsManager.getNameNewPath  (nameOutputExtraLabelsRelPath)

    if (args.rescaleImages):
        rescaleFactorsFileName = joinpathnames(args.datadir, nameRescaleFactors)
        dict_rescaleFactors    = readDictionary(rescaleFactorsFileName)

    if (args.cropImages):
        cropBoundingBoxesFileName = joinpathnames(args.datadir, nameCropBoundingBoxes)
        dict_cropBoundingBoxes    = readDictionary(cropBoundingBoxesFileName)


    # check that number of files is the same in all dirs
    checkNumFilesInDir = [len(listInputImagesFiles), len(listInputLabelsFiles), len(listReferenceFiles)]

    if (args.masksToRegionInterest):
        checkNumFilesInDir.append( len(listInputRoiMasksFiles) )
    if (args.isInputExtraLabels):
        checkNumFilesInDir.append( len(listInputExtraLabelsFiles) )

    for i in range(1, len(checkNumFilesInDir)):
        if (checkNumFilesInDir[i] != checkNumFilesInDir[0]):
            message = 'num images \'%s\' and labels \'%s\' not equal...' %(checkNumFilesInDir[i], checkNumFilesInDir[0])
            CatchErrorException(message)
    #endfor



    for i, (in_image_file, in_label_file) in enumerate(zip(listInputImagesFiles, listInputLabelsFiles)):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))
        print("And: \'%s\'..." % (basename(in_label_file)))

        (inout_image_array, inout_label_array, statusOK) = FileReader.get2ImageArraysAndCheck(in_image_file, in_label_file)
        if not statusOK:
            message = "Images and Labels files not compatible (different size). Skip these data for experiments..."
            CatchWarningException(message)
            continue
        print("Original dims : \'%s\'..." %(str(inout_image_array.shape)))


        if (args.isBinaryTrainMasks):
            print("Convert train masks to binary (0, 1)...")
            inout_label_array = OperationBinaryMasks.process_masks(inout_label_array)


        if (args.isInputExtraLabels):
            in_extralabel_file = listInputExtraLabelsFiles[i]
            print("Process extra train masks: centrelines: \'%s\'..." %(basename(in_extralabel_file)))

            inout_extralabel_array = FileReader.getImageArray(in_extralabel_file)
            inout_extralabel_array = OperationBinaryMasks.process_masks(inout_extralabel_array)


        if (args.masksToRegionInterest):
            in_roimask_file = listInputRoiMasksFiles[i]
            print("Mask train labels to RoI: lungs:  \'%s\'..." %(basename(in_roimask_file)))

            in_roimask_array = FileReader.getImageArray(in_roimask_file)
            in_roimask_array = OperationBinaryMasks.process_masks(in_roimask_array)


        #*******************************************************************************
        if (args.rescaleImages):
            in_reference_file = listReferenceFiles[i]
            rescale_factor = dict_rescaleFactors[filenamenoextension(in_reference_file)]
            print("Rescale image with a factor: \'%s\'..." %(str(rescale_factor)))

            inout_image_array = RescaleImages.compute3D(inout_image_array, rescale_factor, order=args.orderInterpRescale)
            inout_label_array = RescaleImages.compute3D(inout_label_array, rescale_factor, order=args.orderInterpRescale,
                                                        is_binary_mask=True)

            if (args.isInputExtraLabels):
                inout_extralabel_array = RescaleImages.compute3D(inout_extralabel_array, rescale_factor, order=args.orderInterpRescale,
                                                                 is_binary_mask=True)
            if (args.masksToRegionInterest):
                in_roimask_array = RescaleImages.compute3D(in_roimask_array, rescale_factor, order=args.orderInterpRescale,
                                                           is_binary_mask=True)
                # remove noise in masks due to interpolation
                in_roimask_array = ThresholdImages.compute(in_roimask_array, thres_val=0.5)

            print("Final dims: %s..." %(str(inout_image_array.shape)))
        # *******************************************************************************


        # *******************************************************************************
        if (args.masksToRegionInterest):
            print("Apply now the masking of labels...:")
            inout_label_array = OperationBinaryMasks.apply_mask_exclude_voxels(inout_label_array, in_roimask_array)

            if (args.isInputExtraLabels):
                inout_extralabel_array = OperationBinaryMasks.apply_mask_exclude_voxels(inout_extralabel_array, in_roimask_array)
        # *******************************************************************************


        # *******************************************************************************
        if (args.cropImages):
            in_reference_file = listReferenceFiles[i]
            crop_bounding_box = dict_cropBoundingBoxes[filenamenoextension(in_reference_file)]
            print("Crop image to bounding-box: \'%s\'..." % (str(crop_bounding_box)))

            inout_image_array = CropImages.compute3D(inout_image_array, crop_bounding_box)
            inout_label_array = CropImages.compute3D(inout_label_array, crop_bounding_box)

            if (args.isInputExtraLabels):
                inout_extralabel_array = CropImages.compute3D(inout_extralabel_array, crop_bounding_box)

            print("Final dims: %s..." %(str(inout_image_array.shape)))
        # *******************************************************************************


        out_image_file = joinpathnames(OutputImagesPath, nameOutputImagesFiles %(i+1))
        out_label_file = joinpathnames(OutputLabelsPath, nameOutputLabelsFiles %(i+1))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_image_file), (inout_image_array.shape)))
        print("And: \'%s\', of dims \'%s\'..." % (basename(out_label_file), str(inout_label_array.shape)))

        FileReader.writeImageArray(out_image_file, inout_image_array)
        FileReader.writeImageArray(out_label_file, inout_label_array)

        if (args.isInputExtraLabels):
            out_extralabel_file = joinpathnames(OutputExtraLabelsPath, nameOutputExtraLabelsFiles %(i+1))
            print("And: \'%s\', of dims \'%s\'..." % (basename(out_extralabel_file), str(inout_extralabel_array.shape)))

            FileReader.writeImageArray(out_extralabel_file, inout_extralabel_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--isInputExtraLabels', type=str2bool, default=False)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--isBinaryTrainMasks', type=str, default=ISBINARYTRAINMASKS)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    parser.add_argument('--rescaleImages', type=str2bool, default=RESCALEIMAGES)
    parser.add_argument('--orderInterpRescale', type=int, default=ORDERINTERPRESCALE)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in sorted(vars(args).iteritems()):
        print("\'%s\' = %s" %(key, value))

    main(args)