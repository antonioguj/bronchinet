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
    nameInputImagesRelPath       = 'RawAirways/'
    nameInputRoiMasksRelPath     = 'RawLungs/'
    nameReferFilesRelPath        = 'RawImages/'
    nameOutputImagesRelPath      = 'Airways_Rescaled_Order3_Proc/'
    nameOutputMaskedImagesRelPath= 'Airways_Rescaled_Order3_Masked_Proc/'
    nameOutputRoiMasksRelPath    = 'Lungs_Rescaled_Order3_Proc/'
    nameInputImagesFiles         = '*'+ args.extfiles
    nameInputRoiMasksFiles       = '*'+ args.extfiles
    nameInputReferFiles          = '*.dcm'
    nameRescaleFactors           = 'rescaleFactors_images.npy'

    def nameOutputImagesFiles(in_name):
        in_name = in_name.replace('surface0','lumen')
        in_name = in_name.replace('surface1','outwall')
        return filenamenoextension(in_name) + '.nii.gz'

    if args.isInputCentrelines:
        nameInputImagesRelPath  = 'RawCentrelines/'
        nameOutputImagesRelPath = 'Centrelines_Proc/'
        nameOutputMaskedImagesRelPath = 'Centrelines_Masked_Proc/'

        def nameOutputImagesFiles(in_name):
            in_name = in_name.replace('-centrelines','_centrelines')
            return filenamenoextension(in_name) + '.nii.gz'

    nameOutputRoiMasksFiles = lambda in_name: filenamenoextension(in_name).replace('-lungs','_lungs') + '.nii.gz'
    nameOutputMaskedImagesFiles = lambda in_name: filenamenoextension(nameOutputImagesFiles(in_name)) + '_maskedLungs.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager  = WorkDirsManager(args.datadir)
    InputImagesPath  = workDirsManager.getNameExistPath(nameInputImagesRelPath)
    OutputImagesPath = workDirsManager.getNameNewPath  (nameOutputImagesRelPath)

    listInputImagesFiles = findFilesDirAndCheck(InputImagesPath, nameInputImagesFiles)

    if (args.masksToRegionInterest):
        InputRoiMasksPath      = workDirsManager.getNameExistPath(nameInputRoiMasksRelPath)
        OutputRoiMasksPath     = workDirsManager.getNameNewPath  (nameOutputRoiMasksRelPath)
        OutputMaskedImagesPath = workDirsManager.getNameNewPath  (nameOutputMaskedImagesRelPath)

        listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath, nameInputRoiMasksFiles)

    if (args.rescaleImages):
        InputReferFilesPath = workDirsManager.getNameExistPath(nameReferFilesRelPath)
        listInputReferFiles = findFilesDirAndCheck(InputReferFilesPath, nameInputReferFiles)

        rescaleFactorsFileName = joinpathnames(args.datadir, nameRescaleFactors)
        dict_rescaleFactors = readDictionary(rescaleFactorsFileName)



    for i, in_image_file in enumerate(listInputImagesFiles):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))

        in_image_array = FileReader.getImageArray(in_image_file)

        if (args.isClassificationData):
            print("Convert to binary masks (0, 1)...")
            out_image_array = OperationBinaryMasks.process_masks(in_image_array)
        else:
            out_image_array = in_image_array


        if (args.masksToRegionInterest):
            print("Mask input to RoI: lungs...")
            in_roimask_file = listInputRoiMasksFiles[i]
            print("RoI mask (lungs) file: \'%s\'..." %(basename(in_roimask_file)))

            in_roimask_array = FileReader.getImageArray(in_roimask_file)

            if (args.isClassificationData):
                print("Convert to binary masks (0, 1)...")
                out_roimask_array = OperationBinaryMasks.process_masks(in_roimask_array)
            else:
                out_roimask_array = in_roimask_array

            # Image masked to RoI: exclude voxels not contained in lungs
            out_maskedimage_array = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(out_image_array, out_roimask_array)


        if (args.rescaleImages):
            in_refer_file = listInputReferFiles[i]
            rescale_factor = dict_rescaleFactors[filenamenoextension(in_refer_file)]
            print("Rescale image with a factor: \'%s\'..." %(str(rescale_factor)))

            try:
                out_image_array = RescaleImages.compute3D(out_image_array, rescale_factor, is_binary_mask=True)
                print("Final dims: %s..." %(str(out_image_array.shape)))
            except:
                message = 'Issue found when rescaling image'
                CatchWarningException(message)
                continue

            if (args.masksToRegionInterest):
                try:
                    out_roimask_array = RescaleImages.compute3D(out_roimask_array, rescale_factor, is_binary_mask=True)
                except:
                    message = 'Issue found when rescaling image'
                    CatchWarningException(message)
                    continue

                # recompute Image masked to RoI
                out_maskedimage_array = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(out_image_array, out_roimask_array)


        out_file = joinpathnames(OutputImagesPath, nameOutputImagesFiles(basename(in_image_file)))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_file), str(out_image_array.shape)))

        FileReader.writeImageArray(out_file, out_image_array)

        if (args.masksToRegionInterest):
            out_roimask_file = joinpathnames(OutputRoiMasksPath, nameOutputRoiMasksFiles(basename(in_roimask_file)))
            out_maskedimage_file = joinpathnames(OutputMaskedImagesPath, nameOutputMaskedImagesFiles(basename(in_image_file)))

            FileReader.writeImageArray(out_roimask_file, out_roimask_array)
            FileReader.writeImageArray(out_maskedimage_file, out_maskedimage_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--extfiles', type=str, default='.dcm')
    parser.add_argument('--isInputCentrelines', type=str2bool, default=False)
    parser.add_argument('--isClassificationData', type=str, default=ISCLASSIFICATIONDATA)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--rescaleImages', type=str2bool, default=RESCALEIMAGES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
