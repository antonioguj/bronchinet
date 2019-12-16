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
from OperationImages.OperationMasks import *
from collections import OrderedDict
import argparse


LIST_OPERATIONS = ['mask', 'binary_mask', 'crop', 'rescale', 'rescale_mask',
                   'fillholes', 'erode', 'dilate', 'moropen', 'morclose',
                   'thinning', 'threshold', 'normalize', 'power', 'exponential']
DICT_OPERS_SUFFIX = {'mask': 'masked', 'binary_mask': None, 'crop': 'cropped', 'rescale': 'rescaled', 'rescale_mask': 'rescaled',
                     'fillholes': 'fillholes', 'erode': 'eroded', 'dilate': 'dilated', 'moropen': 'moropen', 'morclose': 'morclose',
                     'thinning': 'cenlines', 'threshold': 'binmask', 'normalize': 'normal', 'power': 'power', 'exponential': 'expon'}


def prepare_mask_operation(args):
    print("Operation: Mask images...")
    listInputRoiMasksFiles = findFilesDirAndCheck(args.inputRoidir)

    def wrapfun_mask_image(in_array, i):
        in_roimask_file = listInputRoiMasksFiles[i]
        in_roimask_array = FileReader.getImageArray(in_roimask_file)
        print("Mask to RoI: lungs:  \'%s\'..." % (basename(in_roimask_file)))

        return OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(in_array, in_roimask_array)

    return wrapfun_mask_image


def prepare_binary_mask_operation(args):
    print("Operation: Binarise masks between (0, 1)...")

    def wrapfun_binary_mask_image(in_array, i):
        print("Convert masks to binary (0, 1)...")
        return OperationBinaryMasks.process_masks(in_array)

    return wrapfun_binary_mask_image


def prepare_crop_operation(args):
    print("Operation: Crop images...")
    listReferenceFiles = findFilesDirAndCheck(args.referencedir)
    dict_cropBoundingBoxes = readDictionary(args.boundboxfile)

    def wrapfun_crop_image(in_array, i):
        in_reference_file = listReferenceFiles[i]
        crop_bounding_box = dict_cropBoundingBoxes[filenamenoextension(in_reference_file)]
        print("Crop to bounding-box: \'%s\'..." % (str(crop_bounding_box)))

        return CropImages.compute3D(in_array, crop_bounding_box)

    return wrapfun_crop_image


def prepare_rescale_operation(args, is_rescale_mask= False):
    print("Operation: Rescale images...")
    listReferenceFiles = findFilesDirAndCheck(args.referencedir)
    dict_rescaleFactors = readDictionary(args.rescalefile)

    def wrapfun_rescale_image(in_array, i):
        in_reference_file = listReferenceFiles[i]
        rescale_factor = dict_rescaleFactors[filenamenoextension(in_reference_file)]

        if rescale_factor != (1.0, 1.0, 1.0):
            print("Rescale with a factor: \'%s\'..." % (str(rescale_factor)))
            if is_rescale_mask:
                out_array = RescaleImages.compute3D(in_array, rescale_factor, order=3, is_binary_mask=True)
                # remove noise due to interpolation
                thres_remove_noise = 0.1
                print("Remove noise in mask by thresholding with value: \'%s\'..." %(thres_remove_noise))
                return ThresholdImages.compute(out_array, thres_val=thres_remove_noise)
            else:
                return RescaleImages.compute3D(in_array, rescale_factor, order=3)
        else:
            print("Rescale factor (\'%s\'). Skip rescaling..." % (str(rescale_factor)))
            return in_array

    return wrapfun_rescale_image


def prepare_fillholes_operation(args):
    print("Operation: Fill holes in images...")

    def wrapfun_fillholes_image(in_array, i):
        print("Filling holes from masks...")

        return MorphoFillHolesImages.compute(in_array)

    return wrapfun_fillholes_image


def prepare_erode_operation(args):
    print("Operation: Erode images...")

    def wrapfun_erode_image(in_array, i):
        print("Eroding masks one layer...")

        return MorphoErodeImages.compute(in_array)

    return wrapfun_erode_image


def prepare_dilate_operation(args):
    print("Operation: Dilate images...")

    def wrapfun_dilate_image(in_array, i):
        print("Dilating masks one layer...")

        return MorphoDilateImages.compute(in_array)

    return wrapfun_dilate_image


def prepare_moropen_operation(args):
    print("Operation: Morphologically open images...")

    def wrapfun_moropen_image(in_array, i):
        print("Morphologically open masks...")

        return MorphoOpenImages.compute(in_array)

    return wrapfun_moropen_image


def prepare_morclose_operation(args):
    print("Operation: Morphologically close images...")

    def wrapfun_morclose_image(in_array, i):
        print("Morphologically close masks...")

        return MorphoCloseImages.compute(in_array)

    return wrapfun_morclose_image


def prepare_thinning_operation(args):
    print("Operation: Thinning images to centrelines...")

    def wrapfun_thinning_image(in_array, i):
        print("Thinning masks to extract centrelines...")

        return ThinningMasks.compute(in_array)

    return wrapfun_thinning_image


def prepare_threshold_operation(args):
    print("Operation: Threshold images...")
    thres_value = 0.5

    def wrapfun_threshold_image(in_array, i):
        print("Threshold masks to value \'%s\'..." %(thres_value))

        return ThresholdImages.compute(in_array, thres_val=thres_value)

    return wrapfun_threshold_image


def prepare_normalize_operation(args):
    print("Operation: Normalize images...")

    def wrapfun_normalize_image(in_array, i):
        print("Normalize image to (0,1)...")

        return NormalizeImages.compute3D(in_array)

    return wrapfun_normalize_image


def prepare_power_operation(args):
    print("Operation: Power of images...")
    poworder = 2

    def wrapfun_power_image(in_array, i):
        print("Compute power \'%s\' of image..." % (poworder))
        return np.power(in_array, poworder)

    return wrapfun_power_image


def prepare_exponential_operation(args):
    print("Operation: Exponential of images...")

    def wrapfun_exponential_image(in_array, i):
        print("Compute exponential of image...")
        return (np.exp(in_array) - 1.0) / (np.exp(1) - 1.0)

    return wrapfun_exponential_image




def main(args):

    listInputFiles  = findFilesDirAndCheck(args.inputdir)
    list_names_operations = args.type
    makedir(args.outputdir)


    dict_func_operations = OrderedDict()
    suffix_output_names  = ''

    for name_operation in list_names_operations:
        if name_operation not in LIST_OPERATIONS:
            message = 'Operation \'%s\' not yet implemented...' %(name_operation)
            CatchErrorException(message)
        else:
            if name_operation == 'mask':
                new_func_operation = prepare_mask_operation(args)
            elif name_operation == 'binary_mask':
                new_func_operation = prepare_binary_mask_operation(args)
            elif name_operation == 'crop':
                new_func_operation = prepare_crop_operation(args)
            elif name_operation == 'rescale':
                new_func_operation = prepare_rescale_operation(args)
            elif name_operation == 'rescale_mask':
                new_func_operation = prepare_rescale_operation(args, is_rescale_mask=True)
            elif name_operation == 'fillholes':
                new_func_operation = prepare_fillholes_operation(args)
            elif name_operation == 'erode':
                new_func_operation = prepare_erode_operation(args)
            elif name_operation == 'dilate':
                new_func_operation = prepare_dilate_operation(args)
            elif name_operation == 'moropen':
                new_func_operation = prepare_moropen_operation(args)
            elif name_operation == 'morclose':
                new_func_operation = prepare_morclose_operation(args)
            elif name_operation == 'thinning':
                new_func_operation = prepare_thinning_operation(args)
            elif name_operation == 'threshold':
                new_func_operation = prepare_threshold_operation(args)
            elif name_operation == 'normalize':
                new_func_operation = prepare_normalize_operation(args)
            elif name_operation == 'power':
                new_func_operation = prepare_power_operation(args)
            elif name_operation == 'exponential':
                new_func_operation = prepare_exponential_operation(args)
            else:
                new_func_operation = None
            dict_func_operations[new_func_operation] = new_func_operation

        if DICT_OPERS_SUFFIX[name_operation]:
            suffix_output_names += '_'+ DICT_OPERS_SUFFIX[name_operation]
    #endfor

    nameOutputFiles = lambda in_name: filenamenoextension(in_name) + suffix_output_names +'.nii.gz'



    for i, in_file in enumerate(listInputFiles):
        print("\nInput: \'%s\'..." % (basename(in_file)))

        inout_array = FileReader.getImageArray(in_file)

        for func_name, func_operation in dict_func_operations.items():
            inout_array = func_operation(inout_array, i)  #perform whichever operation
        #endfor


        out_filename = joinpathnames(args.outputdir, nameOutputFiles(basename(in_file)))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_filename), str(inout_array.shape)))

        FileReader.writeImageArray(out_filename, inout_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('inputdir', type=str)
    parser.add_argument('outputdir', type=str)
    parser.add_argument('--type', nargs='+', default=['None'])
    parser.add_argument('--inputRoidir', type=str, default=None)
    parser.add_argument('--referencedir', type=str, default=None)
    parser.add_argument('--boundboxfile', type=str, default=None)
    parser.add_argument('--rescalefile', type=str, default=None)
    args = parser.parse_args()

    if 'mask' in args.type:
        if not args.inputRoidir:
            message = 'need to set argument \'inputRoidir\''
            CatchErrorException(message)
    elif 'crop' in args.type:
        if not args.referencedir or not args.boundboxfile:
            message = 'need to set arguments \'referencedir\' and \'boundboxfile\''
            CatchErrorException(message)
    elif 'rescale' in args.type or \
        'rescale_mask' in args.type:
        if not args.referencedir or not args.rescalefile:
            message = 'need to set arguments \'referencedir\' and \'rescalefile\''
            CatchErrorException(message)

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)