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

    listInputFiles  = findFilesDirAndCheck(args.inputdir)
    makedir(args.outputdir)


    if args.type == 'mask':
        # *********************************************
        print("Operation: Mask images...")

        listInputRoiMasksFiles = findFilesDirAndCheck(args.inputRoidir)
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_masked.nii.gz'

        def wrapfun_mask_image(in_array, i):
            in_roimask_file = listInputRoiMasksFiles[i]
            in_roimask_array = FileReader.getImageArray(in_roimask_file)
            print("Mask to RoI: lungs:  \'%s\'..." % (basename(in_roimask_file)))

            return OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(in_array, in_roimask_array)

        fun_operation = wrapfun_mask_image
        # *********************************************

    elif args.type == 'crop':
        # *********************************************
        print("Operation: Crop images...")

        listReferenceFiles = findFilesDirAndCheck(args.referencedir)
        dict_cropBoundingBoxes = readDictionary(args.boundboxfile)
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_cropped.nii.gz'

        def wrapfun_crop_image(in_array, i):
            in_reference_file = listReferenceFiles[i]
            crop_bounding_box = dict_cropBoundingBoxes[filenamenoextension(in_reference_file)]
            print("Crop to bounding-box: \'%s\'..." % (str(crop_bounding_box)))

            return CropImages.compute3D(in_array, crop_bounding_box)

        fun_operation = wrapfun_crop_image
        # *********************************************

    elif args.type == 'rescale' or args.type == 'rescale_mask':
        # *********************************************
        print("Operation: Rescale images...")

        listReferenceFiles = findFilesDirAndCheck(args.referencedir)
        dict_rescaleFactors = readDictionary(args.rescalefile)
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_rescaled.nii.gz'

        def wrapfun_rescale_image(in_array, i):
            in_reference_file = listReferenceFiles[i]
            rescale_factor = dict_rescaleFactors[filenamenoextension(in_reference_file)]
            print("Rescale with a factor: \'%s\'..." % (str(rescale_factor)))

            thres_remove_noise = 0.1
            if args.type == 'rescale_mask':
                out_array = RescaleImages.compute3D(in_array, rescale_factor, order=3, is_binary_mask=True)
                # remove noise due to interpolation
                return ThresholdImages.compute(out_array, thres_val=thres_remove_noise)
            else:
                return RescaleImages.compute3D(in_array, rescale_factor, order=3)

        fun_operation = wrapfun_rescale_image
        # *********************************************

    elif args.type == 'fillholes':
        # *********************************************
        print("Operation: Fill holes in images...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_fillholes.nii.gz'

        def wrapfun_fillholes_image(in_array, i):
            print("Filling holes from masks...")

            return MorphoFillHolesImages.compute(in_array)

        fun_operation = wrapfun_fillholes_image
        # *********************************************

    elif args.type == 'erode':
        # *********************************************
        print("Operation: Erode images...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_eroded.nii.gz'

        def wrapfun_erode_image(in_array, i):
            print("Eroding masks one layer...")

            return MorphoErodeImages.compute(in_array)

        fun_operation = wrapfun_erode_image
        # *********************************************

    elif args.type == 'dilate':
        # *********************************************
        print("Operation: Dilate images...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_dilated.nii.gz'

        def wrapfun_dilate_image(in_array, i):
            print("Dilating masks one layer...")

            return MorphoDilateImages.compute(in_array)

        fun_operation = wrapfun_dilate_image
        # *********************************************

    elif args.type == 'moropen':
        # *********************************************
        print("Operation: Morphologically open images...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_moropen.nii.gz'

        def wrapfun_moropen_image(in_array, i):
            print("Morphologically open masks...")

            return MorphoOpenImages.compute(in_array)

        fun_operation = wrapfun_moropen_image
        # *********************************************

    elif args.type == 'morclose':
        # *********************************************
        print("Operation: Morphologically close images...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_morclose.nii.gz'

        def wrapfun_morclose_image(in_array, i):
            print("Morphologically close masks...")

            return MorphoCloseImages.compute(in_array)

        fun_operation = wrapfun_morclose_image
        # *********************************************

    elif args.type == 'thinning':
        # *********************************************
        print("Operation: Thinning images to centrelines...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_cenlines.nii.gz'

        def wrapfun_thinning_image(in_array, i):
            print("Thinning masks to extract centrelines...")

            return ThinningMasks.compute(in_array)

        fun_operation = wrapfun_thinning_image
        # *********************************************

    elif args.type == 'threshold':
        # *********************************************
        print("Operation: Threshold images...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_binmak.nii.gz'
        thres_value = 0.5

        def wrapfun_threshold_image(in_array, i):
            print("Threshold masks to value \'%s\'..." %(thres_value))

            return ThresholdImages.compute(in_array, thres_val=thres_value)

        fun_operation = wrapfun_threshold_image
        # *********************************************

    elif args.type == 'normalize':
        # *********************************************
        print("Operation: Normalize images...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_normal.nii.gz'

        def wrapfun_normalize_image(in_array, i):
            print("Normalize image to (0,1)...")

            return NormalizeImages.compute3D(in_array)

        fun_operation = wrapfun_normalize_image
        # *********************************************

    elif args.type == 'power':
        # *********************************************
        print("Operation: Power of images...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_power2.nii.gz'
        poworder = 2

        def wrapfun_power_image(in_array, i):
            print("Compute power \'%s\' of image..." %(poworder))

            return np.power(in_array, poworder)

        fun_operation = wrapfun_power_image
        # *********************************************

    elif args.type == 'exponential':
        # *********************************************
        print("Operation: Exponential of images...")
        nameOutputFiles = lambda in_name: filenamenoextension(in_name) + '_expon.nii.gz'

        def wrapfun_exponential_image(in_array, i):
            print("Compute exponential of image...")

            return (np.exp(in_array) - 1.0) / (np.exp(1) - 1.0)

        fun_operation = wrapfun_exponential_image
        # *********************************************

    else:
        message = 'type operation \'%s\' not found' %(args.type)
        CatchErrorException(message)



    for i, in_file in enumerate(listInputFiles):
        print("\nInput: \'%s\'..." % (basename(in_file)))

        in_array = FileReader.getImageArray(in_file)

        out_array = fun_operation(in_array, i)  #perform whatever operation

        out_file = joinpathnames(args.outputdir, nameOutputFiles(basename(in_file)))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_file), str(out_array.shape)))

        FileReader.writeImageArray(out_file, out_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('inputdir', type=str)
    parser.add_argument('outputdir', type=str)
    parser.add_argument('--type', type=str, default='None')
    parser.add_argument('--inputRoidir', type=str, default=None)
    parser.add_argument('--referencedir', type=str, default=None)
    parser.add_argument('--boundboxfile', type=str, default=None)
    parser.add_argument('--rescalefile', type=str, default=None)
    args = parser.parse_args()

    if args.type == 'mask':
        if not args.inputRoidir:
            message = 'need to set argument \'inputRoidir\''
            CatchErrorException(message)
    elif args.type == 'crop':
        if not args.referencedir or not args.boundboxfile:
            message = 'need to set arguments \'referencedir\' and \'boundboxfile\''
            CatchErrorException(message)
    elif args.type == 'rescale':
        if not args.referencedir or not args.rescalefile:
            message = 'need to set arguments \'referencedir\' and \'rescalefile\''
            CatchErrorException(message)

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)