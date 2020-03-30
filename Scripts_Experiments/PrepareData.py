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


def check_same_number_files_in_list(list_files_1, list_files_2):
    if (len(list_files_1) != len(list_files_2)):
        message = 'num files in two lists not equal: \'%s\' != \'%s\'...' %(len(list_files_1), len(list_files_2))
        CatchErrorException(message)

def check_same_size_arrays(in_array_1, in_array_2):
    if in_array_1.shape != in_array_2.shape:
        message = "arrays have different size: \'%s\' != \'%s\'. Skip these data..." %(str(in_array_1.shape), str(in_array_2.shape))
        CatchWarningException(message)
        return True
    else:
        return False



def main(args):
    # ---------- SETTINGS ----------
    nameTemplateOutputImagesFiles = 'images_proc-%0.2i.nii.gz'
    nameTemplateOutputLabelsFiles = 'labels_proc-%0.2i.nii.gz'
    nameTemplateOutputExtraLabelsFiles = 'cenlines_proc-%0.2i.nii.gz'
    # ---------- SETTINGS ----------



    workDirsManager     = WorkDirsManager(args.datadir)
    InputImagesPath     = workDirsManager.getNameExistPath(args.nameInputImagesRelPath)
    InputReferKeysPath  = workDirsManager.getNameExistPath(args.nameInputReferKeysRelPath)
    OutputImagesPath    = workDirsManager.getNameNewPath  (args.nameOutputImagesRelPath)
    OutputReferKeysFile = workDirsManager.getNameNewUpdateFile(args.nameOutputReferKeysFile)

    listInputImagesFiles    = findFilesDirAndCheck(InputImagesPath)
    listInputReferKeysFiles = findFilesDirAndCheck(InputReferKeysPath)


    if (args.isPrepareLabels):
        InputLabelsPath      = workDirsManager.getNameExistPath(args.nameInputLabelsRelPath)
        OutputLabelsPath     = workDirsManager.getNameNewPath  (args.nameOutputLabelsRelPath)
        listInputLabelsFiles = findFilesDirAndCheck(InputLabelsPath)

        check_same_number_files_in_list(listInputImagesFiles, listInputLabelsFiles)

    if (args.masksToRegionInterest):
        InputRoiMasksPath      = workDirsManager.getNameExistPath(args.nameInputRoiMasksRelPath)
        listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath)

        check_same_number_files_in_list(listInputImagesFiles, listInputRoiMasksFiles)

    if (args.isInputExtraLabels):
        InputExtraLabelsPath      = workDirsManager.getNameExistPath(args.nameInputExtraLabelsRelPath)
        OutputExtraLabelsPath     = workDirsManager.getNameNewPath  (args.nameOutputExtraLabelsRelPath)
        listInputExtraLabelsFiles = findFilesDirAndCheck(InputExtraLabelsPath)

        check_same_number_files_in_list(listInputImagesFiles, listInputExtraLabelsFiles)

    if (args.rescaleImages):
        InputRescaleFactorsFile = workDirsManager.getNameExistPath(args.nameRescaleFactorsFile)
        in_dictRescaleFactors   = readDictionary(InputRescaleFactorsFile)

    if (args.cropImages):
        InputCropBoundingBoxesFile= workDirsManager.getNameExistPath(args.nameCropBoundingBoxesFile)
        in_dictCropBoundingBoxes  = readDictionary(InputCropBoundingBoxesFile)


    if (args.cropImages):
        first_elem_dictCropBoundingBoxes = in_dictCropBoundingBoxes.values()[0]
        if type(first_elem_dictCropBoundingBoxes) == list:
            is_output_multiple_files_per_image = True
            print("\nFound list of crop bounding-boxes per Raw image. Output several processed images...")
            nameTemplateOutputImagesFiles     = 'images_proc-%0.2i_crop-%0.2i.nii.gz'
            nameTemplateOutputLabelsFiles     = 'labels_proc-%0.2i_crop-%0.2i.nii.gz'
            nameTemplateOutputExtraLabelsFiles= 'cenlines_proc-%0.2i_crop-%0.2i.nii.gz'

        else:
            # for new developments, store input dict boundary-boxes per raw images as a list. But output only one processed image
            for key, value in in_dictCropBoundingBoxes.iteritems():
                in_dictCropBoundingBoxes[key] = [value]
            #endfor
            is_output_multiple_files_per_image = False
    else:
        is_output_multiple_files_per_image = False



    out_dictReferenceKeys = OrderedDict()

    for i, in_image_file in enumerate(listInputImagesFiles):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))

        inout_image_array = FileReader.getImageArray(in_image_file)
        print("Original dims : \'%s\'..." % (str(inout_image_array.shape)))

        list_inout_arrays = [inout_image_array]
        list_type_inout_arrays = ['image']


        # *******************************************************************************
        if (args.isPrepareLabels):
            in_label_file = listInputLabelsFiles[i]
            print("And Labels: \'%s\'..." % (basename(in_label_file)))

            inout_label_array = FileReader.getImageArray(in_label_file)
            if (args.isBinaryTrainMasks):
                print("Convert masks to binary (0, 1)...")
                inout_label_array = OperationBinaryMasks.process_masks(inout_label_array)

            list_inout_arrays.append(inout_label_array)
            list_type_inout_arrays.append('label')

            if check_same_number_files_in_list(inout_label_array, inout_image_array):
                continue


        if (args.masksToRegionInterest):
            in_roimask_file = listInputRoiMasksFiles[i]
            print("And ROI Mask for labels: \'%s\'..." %(basename(in_roimask_file)))

            in_roimask_array = FileReader.getImageArray(in_roimask_file)
            if args.isROIlabelsMultiROImasks:
                in_list_roimask_array = OperationBinaryMasks.get_list_masks_all_labels(in_roimask_array)
                list_inout_arrays += in_list_roimask_array
                list_type_inout_arrays += ['roimask'] * len(in_list_roimask_array)
            else:
                in_roimask_array = OperationBinaryMasks.process_masks(in_roimask_array)
                list_inout_arrays.append(in_roimask_array)
                list_type_inout_arrays.append('roimask')

            if check_same_number_files_in_list(in_roimask_array, inout_image_array):
                continue


        if (args.isInputExtraLabels):
            in_extralabel_file = listInputExtraLabelsFiles[i]
            print("And extra labels: \'%s\'..." %(basename(in_extralabel_file)))

            inout_extralabel_array = FileReader.getImageArray(in_extralabel_file)
            inout_extralabel_array = OperationBinaryMasks.process_masks(inout_extralabel_array)
            list_inout_arrays.append(inout_extralabel_array)
            list_type_inout_arrays.append('label')

            if check_same_number_files_in_list(inout_extralabel_array, inout_image_array):
                continue

        num_init_labels = list_type_inout_arrays.count('label')
        # *******************************************************************************


        #*******************************************************************************
        if (args.rescaleImages):
            in_referkey_file = listInputReferKeysFiles[i]
            in_rescale_factor = in_dictRescaleFactors[basenameNoextension(in_referkey_file)]
            print("Rescale image with a factor: \'%s\'..." %(str(in_rescale_factor)))

            if in_rescale_factor != (1.0, 1.0, 1.0):
                for j, (in_array, type_in_array) in enumerate(zip(list_inout_arrays, list_type_inout_arrays)):
                    print('Rescale input array \'%s\' of type \'%s\'...' %(j, type_in_array))
                    if type_in_array == 'image':
                        out_array = RescaleImages.compute3D(in_array, in_rescale_factor, order=3)
                    elif type_in_array == 'label':
                        out_array = RescaleImages.compute3D(in_array, in_rescale_factor, order=3, is_inlabels=True)
                    elif type_in_array == 'roimask':
                        out_array = RescaleImages.compute3D(in_array, in_rescale_factor, order=3, is_inlabels=True, is_binarise_output=True)
                    list_inout_arrays[j] = out_array
                # endfor
            else:
                print("Rescale factor (\'%s'\). Skip rescaling..." %(str(in_rescale_factor)))

            print("Final dims: %s..." %(str(list_inout_arrays[0].shape)))
        # *******************************************************************************


        # *******************************************************************************
        if (args.masksToRegionInterest):
            list_in_calc_arrays    = [list_inout_arrays[j] for j, type_array in enumerate(list_type_inout_arrays) if type_array == 'label']
            list_in_roimask_arrays = [list_inout_arrays[j] for j, type_array in enumerate(list_type_inout_arrays) if type_array == 'roimask']

            list_inout_arrays = [list_inout_arrays[0]]
            list_type_inout_arrays = ['image']

            for j, in_roimask_array in enumerate(list_in_roimask_arrays):
                for k, in_array in enumerate(list_in_calc_arrays):
                    print('Masks input labels array \'%s\' to ROI masks \'%s\'...' % (k, j))
                    out_array = OperationBinaryMasks.apply_mask_exclude_voxels(in_array, in_roimask_array)
                    list_inout_arrays.append(out_array)
                    list_type_inout_arrays.append('label')
                # endfor
            # endfor
        # *******************************************************************************


        # *******************************************************************************
        if (args.cropImages):
            in_referkey_file = listInputReferKeysFiles[i]
            list_in_crop_bounding_boxes = in_dictCropBoundingBoxes[basenameNoextension(in_referkey_file)]
            num_crop_bounding_boxes = len(list_in_crop_bounding_boxes)
            print("Compute \'%s\' cropped images for this raw image:" %(num_crop_bounding_boxes))

            if (args.masksToRegionInterest and args.isROIlabelsMultiROImasks):
                num_total_labels = list_type_inout_arrays.count('label')
                num_total_labels_tocrop = num_crop_bounding_boxes * num_init_labels
                if (num_total_labels_tocrop != num_total_labels):
                    message = 'num labels to crop to bounding boxes is wrong: \'%s\' != \'%s\' (expected)...' %(num_total_labels, num_total_labels_tocrop)
                    CatchErrorException(message)

                # In this set-up, there is already computed the input ROI-masked label per cropping bounding-box
                # Insert input image in the right place: before each set of ROI-masked labels
                in_image_array = list_inout_arrays[0]
                for j in range(1,num_crop_bounding_boxes):
                    pos_insert = j*(num_init_labels+1)
                    list_inout_arrays.insert(pos_insert, in_image_array)
                    list_type_inout_arrays.insert(pos_insert, 'image')
                # endfor
            else:
                # Concatenate as many input images and labels as num cropping bounding-boxes
                list_inout_arrays = list_inout_arrays * num_crop_bounding_boxes
                list_type_inout_arrays = list_type_inout_arrays * num_crop_bounding_boxes


            num_arrays_per_crop_bounding_box = len(list_inout_arrays) // num_crop_bounding_boxes

            icount = 0
            for j, in_crop_bounding_box in enumerate(list_in_crop_bounding_boxes):
                print("Crop input arrays to bounding-box \'%s\' out of total \'%s\': \'%s\'..." % (j, num_crop_bounding_boxes, str(in_crop_bounding_box)))

                size_in_array = list_inout_arrays[icount].shape
                size_in_crop_bounding_box = BoundingBoxes.get_size_bounding_box(in_crop_bounding_box)

                if not BoundingBoxes.is_bounding_box_contained_in_image_size(in_crop_bounding_box, size_in_array):
                    print("Bounding-box is larger than image size: : \'%s\' > \'%s\'. Combine cropping with extending images..."
                          %(str(size_in_crop_bounding_box), str(size_in_array)))

                    new_size_in_array = size_in_crop_bounding_box
                    (croppartial_bounding_box, extendimg_bounding_box) = BoundingBoxes.compute_bounding_boxes_crop_extend_image(in_crop_bounding_box, size_in_array)

                    for k in range(num_arrays_per_crop_bounding_box):
                        print('Crop input array \'%s\' of type \'%s\'...' % (icount, list_type_inout_arrays[icount]))
                        out_array = CropAndExtendImages.compute3D(list_inout_arrays[icount], croppartial_bounding_box,
                                                                  extendimg_bounding_box, new_size_in_array)
                        list_inout_arrays[icount] = out_array
                        icount += 1
                    # endfor
                else:
                    for k in range(num_arrays_per_crop_bounding_box):
                        print('Crop input array \'%s\' of type \'%s\'...' % (icount, list_type_inout_arrays[icount]))
                        out_array = CropImages.compute3D(list_inout_arrays[icount], in_crop_bounding_box)
                        list_inout_arrays[icount] = out_array
                        icount += 1
                    # endfor
            # endfor

            print("Final dims: %s..." % (str(list_inout_arrays[0].shape)))
        # *******************************************************************************


        # Output processed images
        # *******************************************************************************
        if (args.cropImages):
            num_output_files_per_image = num_crop_bounding_boxes
        else:
            num_output_files_per_image = 1

        icount = 0
        for j in range(num_output_files_per_image):
            if is_output_multiple_files_per_image:
                output_image_file = joinpathnames(OutputImagesPath, nameTemplateOutputImagesFiles %(i+1, j+1))
            else:
                output_image_file = joinpathnames(OutputImagesPath, nameTemplateOutputImagesFiles %(i+1))
            print("Output \'%s\' image, of type \'%s\': \'%s\'..." % (icount, list_type_inout_arrays[icount],
                                                                      basename(output_image_file)))

            FileReader.writeImageArray(output_image_file, list_inout_arrays[icount])
            icount += 1

            # save this image in reference keys
            out_dictReferenceKeys[basenameNoextension(output_image_file)] = basename(in_image_file)


            if (args.isPrepareLabels):
                if is_output_multiple_files_per_image:
                    output_label_file = joinpathnames(OutputLabelsPath, nameTemplateOutputLabelsFiles % (i+1, j+1))
                else:
                    output_label_file = joinpathnames(OutputLabelsPath, nameTemplateOutputLabelsFiles % (i+1))
                print("Output \'%s\' label, of type \'%s\': \'%s\'..." % (icount, list_type_inout_arrays[icount],
                                                                          basename(output_label_file)))

                FileReader.writeImageArray(output_label_file, list_inout_arrays[icount])
                icount += 1

            if (args.isInputExtraLabels):
                if is_output_multiple_files_per_image:
                    output_extralabel_file = joinpathnames(OutputLabelsPath, nameTemplateOutputLabelsFiles % (i+1, j+1))
                else:
                    output_extralabel_file = joinpathnames(OutputLabelsPath, nameTemplateOutputLabelsFiles % (i+1))
                print("Output \'%s\' extra label, of type \'%s\': \'%s\'..." % (icount, list_type_inout_arrays[icount],
                                                                                basename(output_extralabel_file)))

                FileReader.writeImageArray(output_extralabel_file, list_inout_arrays[icount])
                icount += 1
        # endfor
        # *******************************************************************************
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
    parser.add_argument('--nameOutputReferKeysFile', type=str, default=NAME_REFERKEYSPROCIMAGE_FILE)
    parser.add_argument('--isPrepareLabels', type=str2bool, default=True)
    parser.add_argument('--isInputExtraLabels', type=str2bool, default=False)
    parser.add_argument('--isBinaryTrainMasks', type=str2bool, default=ISBINARYTRAINMASKS)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--rescaleImages', type=str2bool, default=RESCALEIMAGES)
    parser.add_argument('--nameRescaleFactorsFile', type=str, default=NAME_RESCALEFACTOR_FILE)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    parser.add_argument('--nameCropBoundingBoxesFile', type=str, default=NAME_CROPBOUNDINGBOX_FILE)
    parser.add_argument('--isROIlabelsMultiROImasks', type=str2bool, default=ISTWOBOUNDINGBOXEACHLUNGS)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in sorted(vars(args).iteritems()):
        print("\'%s\' = %s" %(key, value))

    main(args)