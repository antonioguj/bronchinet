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
    InputPath     = args.inputdir
    OutputPath    = args.outputdir
    ExtraInputPath= args.extrainputdir
    # ---------- SETTINGS ----------


    listInputFiles  = findFilesDirAndCheck(InputPath)
    makedir(OutputPath)


    if args.typeOperation == 'mask':
        # *********************************************
        print("Operation: Mask images...")
        if not isExistdir(args.extrainputdir):
            raise Exception("ERROR. input dir \'%s\' not found... EXIT" %(ExtraInputPath))

        listInputRoiMasksFiles = findFilesDirAndCheck(ExtraInputPath)
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_masked.nii.gz'

        def wrapfun_mask_image(in_array, i):
            in_roimask_file = listInputRoiMasksFiles[i]
            in_roimask_array = FileReader.getImageArray(in_roimask_file)
            print("Mask to RoI: lungs:  \'%s\'..." % (basename(in_roimask_file)))

            return OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(in_array, in_roimask_array)

        fun_operation = wrapfun_mask_image
        # *********************************************

    elif args.typeOperation == 'crop':
        # *********************************************
        print("Operation: Crop images...")
        if not isExistdir(args.extrainputdir):
            raise Exception("ERROR. input dir \'%s\' not found... EXIT" %(ExtraInputPath))

        listReferenceFiles = findFilesDirAndCheck(ExtraInputPath)
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_cropped.nii.gz'

        nameCropBoundingBoxes = 'cropBoundingBoxes_images.npy'
        cropBoundingBoxesFileName = joinpathnames(args.datadir, nameCropBoundingBoxes)
        dict_cropBoundingBoxes = readDictionary(cropBoundingBoxesFileName)

        def wrapfun_crop_image(in_array, i):
            in_reference_file = listReferenceFiles[i]
            crop_bounding_box = dict_cropBoundingBoxes[filenamenoextension(in_reference_file)]
            print("Crop to bounding-box: \'%s\'..." % (str(crop_bounding_box)))

            return CropImages.compute3D(in_array, crop_bounding_box)

        fun_operation = wrapfun_crop_image
        # *********************************************

    elif args.typeOperation == 'rescale' or args.typeOperation == 'rescale_mask':
        # *********************************************
        print("Operation: Rescale images...")
        if not isExistdir(args.extrainputdir):
            raise Exception("ERROR. input dir \'%s\' not found... EXIT" %(ExtraInputPath))

        listReferenceFiles = findFilesDirAndCheck(ExtraInputPath)
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_rescaled.nii.gz'

        nameRescaleFactors = 'rescaleFactors_images.npy'
        rescaleFactorsFileName = joinpathnames(args.datadir, nameRescaleFactors)
        dict_rescaleFactors = readDictionary(rescaleFactorsFileName)

        def wrapfun_rescale_image(in_array, i):
            in_reference_file = listReferenceFiles[i]
            rescale_factor = dict_rescaleFactors[filenamenoextension(in_reference_file)]
            print("Rescale with a factor: \'%s\'..." % (str(rescale_factor)))

            if args.typeOperation == 'rescale_mask':
                out_array = RescaleImages.compute3D(in_array, rescale_factor, order=3, is_binary_mask=True)
                # remove noise due to interpolation
                return ThresholdImages.compute(out_array, thres_val=0.5)
            else:
                return RescaleImages.compute3D(in_array, rescale_factor, order=3)

        fun_operation = wrapfun_rescale_image
        # *********************************************

    elif args.typeOperation == 'fill_holes':
        # *********************************************
        print("Operation: Fill holes in images...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_fillholes.nii.gz'

        def wrapfun_fillholes_image(in_array, i):
            print("Filling holes from masks...")

            return FillInHolesImages.compute(in_array)

        fun_operation = wrapfun_fillholes_image
        # *********************************************

    elif args.typeOperation == 'thinning':
        # *********************************************
        print("Operation: Thinning images to centrelines...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_cenlines.nii.gz'

        def wrapfun_thinning_image(in_array, i):
            print("Thinning masks to extract centrelines...")

            return ThinningMasks.compute(in_array)

        fun_operation = wrapfun_thinning_image
        # *********************************************

    elif args.typeOperation == 'threshold':
        # *********************************************
        print("Operation: Threshold images...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_binmak.nii.gz'
        thres_value = 0.5

        def wrapfun_threshold_image(in_array, i):
            print("Threshold masks to value \'%s\'..." %(thres_value))

            return ThresholdImages.compute(in_array, thres_val=thres_value)

        fun_operation = wrapfun_threshold_image
        # *********************************************

    elif args.typeOperation == 'normalize':
        # *********************************************
        print("Operation: Normalize images...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_normal.nii.gz'

        def wrapfun_normalize_image(in_array, i):
            print("Normalize image to (0,1)...")

            return NormalizeImages.compute3D(in_array)

        fun_operation = wrapfun_normalize_image
        # *********************************************

    elif args.typeOperation == 'power_image':
        # *********************************************
        print("Operation: Power of images...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_power2.nii.gz'
        poworder = 2

        def wrapfun_power_image(in_array, i):
            print("Compute power \'%s\' of image..." %(poworder))

            return np.power(in_array, poworder)

        fun_operation = wrapfun_power_image
        # *********************************************

    elif args.typeOperation == 'exponential_image':
        # *********************************************
        print("Operation: Exponential of images...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_expon.nii.gz'

        def wrapfun_exponential_image(in_array, i):
            print("Compute exponential of image...")

            return (np.exp(in_array) - 1.0) / (np.exp(1) - 1.0)

        fun_operation = wrapfun_exponential_image
        # *********************************************

    else:
        raise Exception("ERROR. type operation \'%s\' not found... EXIT" %(args.typeOperation))



    for i, in_file in enumerate(listInputFiles):
        print("\nInput: \'%s\'..." % (basename(in_file)))

        in_array = FileReader.getImageArray(in_file)

        out_array = fun_operation(in_array, i)  #perform whatever operation

        out_file = joinpathnames(OutputPath, nameOutputFiles(basename(in_file)))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_file), str(out_array.shape)))

        FileReader.writeImageArray(out_file, out_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('inputdir', type=str)
    parser.add_argument('outputdir', type=str)
    parser.add_argument('--typeOperation', type=str, default='None')
    parser.add_argument('--extrainputdir', type=str, default='None')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)