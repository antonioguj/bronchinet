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
    nameInputImagesRelPath   = 'Images_Proc/'
    nameInputLabelsRelPath   = 'Airways_Proc/'
    nameInputRoiMasksRelPath = 'Lungs_Proc/'
    nameOutputImagesRelPath  = 'Images_WorkData/'
    nameOutputLabelsRelPath  = 'Labels_WorkData/'
    nameInputImagesFiles     = '*.nii.gz'
    nameInputLabelsFiles     = '*.nii.gz'
    nameInputRoiMasksFiles   = '*.nii.gz'
    nameCropBoundingBoxes    = 'cropBoundingBoxes_images.npy'
    nameRescaleFactors       = 'rescaleFactors_images.npy'
    nameOutputImagesFiles    = 'images-%0.2i_dim%s' + getFileExtension(FORMATTRAINDATA)
    nameOutputLabelsFiles    = 'labels-%0.2i_dim%s' + getFileExtension(FORMATTRAINDATA)

    if (args.saveVisualizeProcData):
        nameVisualOutputRelPath = 'VisualizeWorkData'
        nameOutputVisualFiles   = lambda filename: basename(filename).replace('.npz','.nii.gz')
    # ---------- SETTINGS ----------


    workDirsManager  = WorkDirsManager(args.datadir)
    InputImagesPath  = workDirsManager.getNameExistPath(nameInputImagesRelPath )
    InputLabelsPath  = workDirsManager.getNameExistPath(nameInputLabelsRelPath )
    OutputImagesPath = workDirsManager.getNameNewPath  (nameOutputImagesRelPath)
    OutputLabelsPath = workDirsManager.getNameNewPath  (nameOutputLabelsRelPath)

    listInputImagesFiles = findFilesDirAndCheck(InputImagesPath, nameInputImagesFiles)
    listInputLabelsFiles = findFilesDirAndCheck(InputLabelsPath, nameInputLabelsFiles)

    if (len(listInputImagesFiles) != len(listInputLabelsFiles)):
        message = 'num files in dir 1 \'%s\', not equal to num files in dir 2 \'%i\'...' %(len(listInputImagesFiles),
                                                                                           len(listInputLabelsFiles))
        CatchErrorException(message)

    if (args.masksToRegionInterest):
        InputRoiMasksPath     = workDirsManager.getNameExistPath(nameInputRoiMasksRelPath)
        listInputRoiMaskFiles = findFilesDirAndCheck(InputRoiMasksPath, nameInputRoiMasksFiles)

    if (args.saveVisualizeProcData):
        VisualOutputPath = workDirsManager.getNameNewPath(nameVisualOutputRelPath)

    if (args.rescaleImages):
        dict_rescaleFactors = readDictionary(joinpathnames(args.datadir, nameRescaleFactors))

    if (args.cropImages):
        dict_cropBoundingBoxes = readDictionary(joinpathnames(args.datadir, nameCropBoundingBoxes))



    # START ANALYSIS
    # ------------------------------
    print("-" * 30)
    print("Preprocessing...")
    print("-" * 30)

    for i, (in_image_file, in_label_file) in enumerate(zip(listInputImagesFiles, listInputLabelsFiles)):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))
        print("And: \'%s\'..." % (basename(in_label_file)))

        (image_array, label_array) = FileReader.get2ImageArraysAndCheck(in_image_file, in_label_file)
        print("Original dims : \'%s\'..." %(str(image_array.shape)))

        if (args.isClassificationData):
            print("Convert to binary masks (0, 1)...")
            label_array = OperationBinaryMasks.process_masks(label_array)


        if (args.masksToRegionInterest):
            print("Mask input to RoI: lungs...")
            in_roimask_file = listInputRoiMaskFiles[i]
            print("RoI mask (lungs) file: \'%s\'..." %(basename(in_roimask_file)))

            roimask_array = FileReader.getImageArray(in_roimask_file)
            label_array = OperationBinaryMasks.apply_mask_exclude_voxels(label_array, roimask_array)


        if (args.rescaleImages):
            rescale_factor = dict_rescaleFactors[filenamenoextension(in_image_file)]
            print("Rescale image with a factor: \'%s\'..." %(str(rescale_factor)))

            image_array = RescaleImages.compute3D(image_array, rescale_factor)
            label_array = RescaleImages.compute3D(label_array, rescale_factor)
            print("Final dims: %s..." %(str(image_array.shape)))


        if (args.cropImages):
            crop_bounding_box = dict_cropBoundingBoxes[filenamenoextension(in_image_file)]
            print("Crop image to bounding-box: \'%s\'..." % (str(crop_bounding_box)))

            image_array = CropImages.compute3D(image_array, crop_bounding_box)
            label_array = CropImages.compute3D(label_array, crop_bounding_box)
            print("Final dims: %s..." %(str(image_array.shape)))


        # if (args.extendSizeImages):
        #     print("Extend images to fixed size \'%s\':..." %(str(CROPSIZEBOUNDINGBOX)))
        #     size_new_image = (image_array.shape[0], CROPSIZEBOUNDINGBOX[0], CROPSIZEBOUNDINGBOX[1])
        #     backgr_val_images = -1000
        #     backgr_val_masks = -1 if args.masksToRegionInterest else 0
        #     bounding_box = dict_bounding_boxes[filenamenoextension(in_image_file)]
        #
        #     image_array = ExtendImages.compute3D(image_array, bounding_box, size_new_image, background_value=backgr_val_images)
        #     label_array = ExtendImages.compute3D(label_array, bounding_box, size_new_image, background_value=backgr_val_masks)
        #     print("Final dims: %s..." % (str(image_array.shape)))


        out_image_file = joinpathnames(OutputImagesPath, nameOutputImagesFiles %(i+1, tuple2str(image_array.shape)))
        out_label_file = joinpathnames(OutputLabelsPath, nameOutputLabelsFiles %(i+1, tuple2str(label_array.shape)))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_image_file), (image_array.shape)))
        print("And: \'%s\', of dims \'%s\'..." % (basename(out_label_file), str(label_array.shape)))

        FileReader.writeImageArray(out_image_file, image_array)
        FileReader.writeImageArray(out_label_file, label_array)

        if (args.saveVisualizeProcData):
            print("Saving working data to visualize...")
            out_image_file = joinpathnames(VisualOutputPath, nameOutputVisualFiles(out_image_file))
            out_label_file = joinpathnames(VisualOutputPath, nameOutputVisualFiles(out_label_file))

            FileReader.writeImageArray(out_image_file, image_array)
            FileReader.writeImageArray(out_label_file, label_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=DATADIR)
    parser.add_argument('--isClassificationData', type=str, default=ISCLASSIFICATIONDATA)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--rescaleImages', type=str2bool, default=False)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    parser.add_argument('--extendSizeImages', type=str2bool, default=EXTENDSIZEIMAGES)
    parser.add_argument('--saveVisualizeProcData', type=str2bool, default=SAVEVISUALIZEPROCDATA)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
