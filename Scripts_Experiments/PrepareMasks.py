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
    nameInputImagesRelPath = 'RawAirways'
    nameInputRoiMasksRelPath = 'RawLungs'
    nameReferenceImgRelPath = 'RawImages'
    nameOutputImagesRelPath = 'Airways_Rescaled_0.6x0.6x0.6_Full'
    nameOutputRoiMasksRelPath = 'Lungs_Rescaled_0.6x0.6x0.6_Full'

    nameInputImagesFiles = '*surface0.dcm'
    nameInputRoiMasksFiles = '*.dcm'
    nameReferenceImgFiles = '*.dcm'
    # prefixPatternInputFiles = 'av[0-9][0-9]*'

    nameRescaleFactors = 'rescaleFactors_images_0.6x0.6x0.6.npy'

    def nameOutputImagesFiles(in_name):
        in_name = in_name.replace('surface0','lumen')
        in_name = in_name.replace('surface1','outwall')
        #in_name = in_name.replace('-result','_noopfront')
        #in_name = in_name.replace('-centrelines','_centrelines')
        return filenamenoextension(in_name) + '.nii.gz'

    nameOutputRoiMasksFiles = lambda in_name: filenamenoextension(in_name).replace('-lungs','_lungs') + '.nii.gz'
    nameOutputImagesMaskedToRoiFiles = lambda in_name: filenamenoextension(nameOutputImagesFiles(in_name)) + '_maskedToLungs.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(args.basedir)
    BaseDataPath = workDirsManager.getNameBaseDataPath()
    InputImagesPath = workDirsManager.getNameExistPath(BaseDataPath, nameInputImagesRelPath)
    ReferenceImgPath = workDirsManager.getNameExistPath(BaseDataPath, nameReferenceImgRelPath)
    OutputImagesPath = workDirsManager.getNameNewPath(BaseDataPath, nameOutputImagesRelPath)

    listInputImagesFiles = findFilesDirAndCheck(InputImagesPath, nameInputImagesFiles)
    listReferenceImgFiles = findFilesDirAndCheck(ReferenceImgPath, nameReferenceImgFiles)

    if (args.masksToRegionInterest):
        InputRoiMasksPath = workDirsManager.getNameExistPath(BaseDataPath, nameInputRoiMasksRelPath)
        OutputRoiMasksPath = workDirsManager.getNameNewPath(BaseDataPath, nameOutputRoiMasksRelPath)

        listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath, nameInputRoiMasksFiles)

    if (args.rescaleImages):
        dict_rescaleFactors = readDictionary(joinpathnames(BaseDataPath, nameRescaleFactors))



    for i, in_image_file in enumerate(listInputImagesFiles):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))

        image_array = FileReader.getImageArray(in_image_file)

        if (args.isClassificationData):
            print("Convert to binary masks (0, 1)...")
            image_array = OperationBinaryMasks.process_masks(image_array)


        if (args.masksToRegionInterest):
            print("Mask input to RoI: lungs...")
            in_roimask_file = findFileWithSamePrefix(basename(in_image_file), listInputRoiMasksFiles)
            print("RoI mask (lungs) file: \'%s\'..." %(basename(in_roimask_file)))

            roimask_array = FileReader.getImageArray(in_roimask_file)

            if (args.isClassificationData):
                print("Convert to binary masks (0, 1)...")
                roimask_array = OperationBinaryMasks.process_masks(roimask_array)

            # Image masked to RoI: exclude voxels not contained in lungs
            image_maskedToRoi_array = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(image_array, roimask_array)


        if (args.rescaleImages):
            in_referimg_file = findFileWithSamePrefix(basename(in_image_file), listReferenceImgFiles)
            rescale_factor = dict_rescaleFactors[filenamenoextension(in_referimg_file)]
            print("Rescale image with a factor: \'%s\'..." %(str(rescale_factor)))

            image_array = RescaleImages.compute3D(image_array, rescale_factor, is_binary_mask=True)
            print("Final dims: %s..." %(str(image_array.shape)))

            if (args.masksToRegionInterest):
                roimask_array = RescaleImages.compute3D(roimask_array, rescale_factor, is_binary_mask=True)
                image_maskedToRoi_array = RescaleImages.compute3D(image_maskedToRoi_array, rescale_factor, is_binary_mask=True)


        out_file = joinpathnames(OutputImagesPath, nameOutputImagesFiles(basename(in_image_file)))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_file), str(image_array.shape)))

        FileReader.writeImageArray(out_file, image_array)

        if (args.masksToRegionInterest):
            out_roimask_file = joinpathnames(OutputRoiMasksPath, nameOutputRoiMasksFiles(basename(in_roimask_file)))
            out_maskedToRoi_file = joinpathnames(OutputImagesPath, nameOutputImagesMaskedToRoiFiles(basename(in_image_file)))

            FileReader.writeImageArray(out_roimask_file, roimask_array)
            FileReader.writeImageArray(out_maskedToRoi_file, image_maskedToRoi_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--isClassificationData', type=str, default=ISCLASSIFICATIONDATA)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--rescaleImages', type=str2bool, default=RESCALEIMAGES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
