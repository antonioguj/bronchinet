
from common.constant import DATADIR
from common.functionutil import *
from dataloaders.imagefilereader import ImageFileReader
from imageoperators.imageoperator import CropImage, RescaleImage, MorphoDilateMask, MorphoErodeMask, MorphoOpenMask, \
                                         MorphoCloseMask, MorphoFillHolesMask, ConnectedRegionsMask, FirstConnectedRegionMask, \
                                         ThresholdImage, ThinningMask, NormaliseImage
from imageoperators.maskoperator import MaskOperator
from collections import OrderedDict
import argparse


LIST_OPERATIONS = ['mask', 'binarise', 'merge', 'substract', 'crop', 'rescale', 'rescalemask',
                   'fillholes', 'erode', 'dilate', 'moropen', 'morclose', 'connregs', 'firstconreg',
                   'thinning', 'threshold', 'masklabels', 'normalise', 'power', 'exponential']
DICT_OPERS_SUFFIX = {'mask': 'masked',
                     'binarise': None,
                     'merge': 'merged',
                     'substract': 'substed',
                     'crop': 'cropped',
                     'rescale': 'rescaled',
                     'rescalemask': 'rescaled',
                     'fillholes': 'fillholes',
                     'erode': 'eroded',
                     'dilate': 'dilated',
                     'moropen': 'moropen',
                     'morclose': 'morclose',
                     'connregs': 'connregs',
                     'firstconreg': 'firstreg',
                     'thinning': 'cenlines',
                     'threshold': 'binmask',
                     'masklabels': 'lab%s',
                     'normalise': 'normal',
                     'power': 'power',
                     'exponential': 'expon'}


# ------------------------------------------------
def prepare_mask_operation(args):
    print("Operation: Mask images...")
    listInputRoiMasksFiles = list_files_dir(args.input_roidir)

    def wrapfun_mask_image(in_data, i):
        in_roimask_file = listInputRoiMasksFiles[i]
        in_roimask = ImageFileReader.get_image(in_roimask_file)
        print("Mask to RoI: lungs: \'%s\'..." % (basename(in_roimask_file)))

        return MaskOperator.mask_exclude_regions_fillzero(in_data, in_roimask)

    return wrapfun_mask_image


# ------------------------------------------------
def prepare_binarise_operation(args):
    print("Operation: Binarise masks between (0, 1)...")

    def wrapfun_binarise_image(in_data, i):
        print("Convert masks to binary (0, 1)...")
        return MaskOperator.binarise(in_data)

    return wrapfun_binarise_image


# ------------------------------------------------
def prepare_merge_operation(args):
    print("Operation: Merge two images...")
    listInput2ndMasksFiles = list_files_dir(args.input_2nddir)

    def wrapfun_merge_image(in_data, i):
        in_2ndmask_file = listInput2ndMasksFiles[i]
        in_2ndmask = ImageFileReader.get_image(in_2ndmask_file)
        print("2nd mask file: \'%s\'..." % (basename(in_2ndmask_file)))

        return MaskOperator.merge_two_masks(in_data, in_2ndmask)

    return wrapfun_merge_image


# ------------------------------------------------
def prepare_substract_operation(args):
    print("Operation: Substract two images (img1 - img2)...")
    listInput2ndMasksFiles = list_files_dir(args.input_2nddir)

    def wrapfun_substract_image(in_data, i):
        in_2ndmask_file = listInput2ndMasksFiles[i]
        in_2ndmask = ImageFileReader.get_image(in_2ndmask_file)
        print("2nd mask file: \'%s\'..." % (basename(in_2ndmask_file)))

        return MaskOperator.substract_two_masks(in_data, in_2ndmask)

    return wrapfun_substract_image


# ------------------------------------------------
def prepare_crop_operation(args):
    print("Operation: Crop images...")
    listReferenceFiles = list_files_dir(args.reference_dir)
    dict_cropBoundingBoxes = read_dictionary(args.boundbox_file)

    def wrapfun_crop_image(in_data, i):
        in_referkey_file = listReferenceFiles[i]
        crop_bounding_box = dict_cropBoundingBoxes[basename_file_noext(in_referkey_file)]
        print("Crop to bounding-box: \'%s\'..." % (str(crop_bounding_box)))

        return CropImage._compute3D(in_data, crop_bounding_box)

    return wrapfun_crop_image


# ------------------------------------------------
def prepare_rescale_updatemetadata(args):
    print("Update metadata (voxel size) from Rescaling operation...")
    listReferenceFiles = list_files_dir(args.reference_dir)
    dict_rescaleFactors = read_dictionary(args.rescale_file)

    def wrapfun_updatemetadata_rescale(in_file, i):
        in_referkey_file = listReferenceFiles[i]
        rescale_factor = dict_rescaleFactors[basename_file_noext(in_referkey_file)]
        print("Update metadata for a rescaling factor: \'%s\'..." % (str(rescale_factor)))

        return ImageFileReader.update_image_metadata_info(in_file, rescale_factor=rescale_factor)

    return wrapfun_updatemetadata_rescale


def prepare_rescale_operation(args, is_rescale_mask= False):
    print("Operation: Rescale images...")
    listReferenceFiles = list_files_dir(args.reference_dir)
    dict_rescaleFactors = read_dictionary(args.rescale_file)

    def wrapfun_rescale_image(in_data, i):
        in_referkey_file = listReferenceFiles[i]
        rescale_factor = dict_rescaleFactors[basename_file_noext(in_referkey_file)]

        if rescale_factor != (1.0, 1.0, 1.0):
            print("Rescale with a factor: \'%s\'..." % (str(rescale_factor)))
            if is_rescale_mask:
                return RescaleImage.compute(in_data, rescale_factor, order=3, is_inlabels=True, is_binarise_output=True)
            else:
                return RescaleImage.compute(in_data, rescale_factor, order=3)
        else:
            print("Rescale factor (\'%s\'). Skip rescaling..." % (str(rescale_factor)))
            return in_data

    return wrapfun_rescale_image


# ------------------------------------------------
def prepare_fillholes_operation(args):
    print("Operation: Fill holes in images...")

    def wrapfun_fillholes_image(in_data, i):
        print("Filling holes from masks...")
        return MorphoFillHolesMask.compute(in_data)

    return wrapfun_fillholes_image


# ------------------------------------------------
def prepare_erode_operation(args):
    print("Operation: Erode images...")

    def wrapfun_erode_image(in_data, i):
        print("Eroding masks one layer...")
        return MorphoErodeMask.compute(in_data)

    return wrapfun_erode_image


# ------------------------------------------------
def prepare_dilate_operation(args):
    print("Operation: Dilate images...")

    def wrapfun_dilate_image(in_data, i):
        print("Dilating masks one layer...")
        return MorphoDilateMask.compute(in_data)

    return wrapfun_dilate_image


# ------------------------------------------------
def prepare_moropen_operation(args):
    print("Operation: Morphologically open images...")

    def wrapfun_moropen_image(in_data, i):
        print("Morphologically open masks...")
        return MorphoOpenMask.compute(in_data)

    return wrapfun_moropen_image


# ------------------------------------------------
def prepare_morclose_operation(args):
    print("Operation: Morphologically close images...")

    def wrapfun_morclose_image(in_data, i):
        print("Morphologically close masks...")
        return MorphoCloseMask.compute(in_data)

    return wrapfun_morclose_image


# ------------------------------------------------
def prepare_connregions_operation(args):
    print("Operation: Compute connected regions...")

    def wrapfun_connregions_image(in_data, i):
        print("Compute connected regions...")
        #out_data, num_regions = ConnectedRegionsMasks.compute_get_num_regs(in_data)
        out_data, num_regions = ConnectedRegionsMask.compute_get_num_regs(in_data, connectivity_dim=1)
        print("Number connected regions: \'%s\'..." %(num_regions))
        return out_data

    return wrapfun_connregions_image


# ------------------------------------------------
def prepare_firstconnregion_operation(args):
    print("Operation: Compute the first connected region (with largest volume)...")

    def wrapfun_firstconnregion_image(in_data, i):
        print("Compute the first connected region...")
        #out_data = FirstConnectedRegionMasks.compute(in_data)
        out_data = FirstConnectedRegionMask.compute(in_data, connectivity_dim=1)
        return out_data

    return wrapfun_firstconnregion_image


# ------------------------------------------------
def prepare_thinning_operation(args):
    print("Operation: Thinning images to centrelines...")

    def wrapfun_thinning_image(in_data, i):
        print("Thinning masks to extract centrelines...")
        return ThinningMask.compute(in_data)

    return wrapfun_thinning_image


# ------------------------------------------------
def prepare_threshold_operation(args):
    print("Operation: Threshold images...")
    thres_value = 0.5

    def wrapfun_threshold_image(in_data, i):
        print("Threshold masks to value \'%s\'..." %(thres_value))
        return ThresholdImage.compute(in_data, thres_val=thres_value)

    return wrapfun_threshold_image


# ------------------------------------------------
def prepare_maskwithlabels_operation(args):
    in_mask_labels = args.in_mask_labels
    print("Operation: Retrieve from the mask the labels: \'%s\'..." %(in_mask_labels))

    def wrapfun_maskwithlabels_image(in_data, i):
        print("Retrieve labels: \'%s\'..." %(in_mask_labels))
        return MaskOperator.get_masks_with_labels_list(in_data, inlist_labels=in_mask_labels)

    return wrapfun_maskwithlabels_image


# ------------------------------------------------
def prepare_normalise_operation(args):
    print("Operation: normalise images...")

    def wrapfun_normalise_image(in_data, i):
        print("Normalise image to (0,1)...")
        return NormaliseImage.compute(in_data)

    return wrapfun_normalise_image


# ------------------------------------------------
def prepare_power_operation(args):
    print("Operation: Power of images...")
    poworder = 2

    def wrapfun_power_image(in_data, i):
        print("Compute power \'%s\' of image..." % (poworder))
        return np.power(in_data, poworder)

    return wrapfun_power_image


# ------------------------------------------------
def prepare_exponential_operation(args):
    print("Operation: Exponential of images...")

    def wrapfun_exponential_image(in_data, i):
        print("Compute exponential of image...")
        return (np.exp(in_data) - 1.0) / (np.exp(1) - 1.0)

    return wrapfun_exponential_image




def main(args):

    listInputFiles  = list_files_dir(args.input_dir)
    list_names_operations = args.type
    makedir(args.output_dir)


    dict_func_operations = OrderedDict()
    suffix_output_names  = ''

    is_update_metadata_file = False
    func_updatemetadata = None

    for name_operation in list_names_operations:
        if name_operation not in LIST_OPERATIONS and name_operation != 'None':
            message = 'Operation \'%s\' not yet implemented...' %(name_operation)
            catch_error_exception(message)
        else:
            if name_operation == 'mask':
                new_func_operation = prepare_mask_operation(args)
            elif name_operation == 'binarise':
                new_func_operation = prepare_binarise_operation(args)
            elif name_operation == 'merge':
                new_func_operation = prepare_merge_operation(args)
            elif name_operation == 'substract':
                new_func_operation = prepare_substract_operation(args)
            elif name_operation == 'crop':
                new_func_operation = prepare_crop_operation(args)
            elif name_operation == 'rescale':
                is_update_metadata_file = True
                new_func_operation = prepare_rescale_operation(args)
                func_updatemetadata = prepare_rescale_updatemetadata(args)
            elif name_operation == 'rescalemask':
                is_update_metadata_file = True
                new_func_operation = prepare_rescale_operation(args, is_rescale_mask=True)
                func_updatemetadata = prepare_rescale_updatemetadata(args)
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
            elif name_operation == 'connregs':
                new_func_operation = prepare_connregions_operation(args)
            elif name_operation == 'firstconreg':
                new_func_operation = prepare_firstconnregion_operation(args)
            elif name_operation == 'thinning':
                new_func_operation = prepare_thinning_operation(args)
            elif name_operation == 'threshold':
                new_func_operation = prepare_threshold_operation(args)
            elif name_operation == 'masklabels':
                new_func_operation = prepare_maskwithlabels_operation(args)
                DICT_OPERS_SUFFIX['masklabels'] = DICT_OPERS_SUFFIX['masklabels'] %('-'.join([str(el) for el in args.in_mask_labels]))
            elif name_operation == 'normalise':
                new_func_operation = prepare_normalise_operation(args)
            elif name_operation == 'power':
                new_func_operation = prepare_power_operation(args)
            elif name_operation == 'exponential':
                new_func_operation = prepare_exponential_operation(args)
            elif name_operation == 'None':
                def fun_do_nothing(in_data, i):
                    return in_data
                new_func_operation = fun_do_nothing
            else:
                new_func_operation = None
            dict_func_operations[name_operation] = new_func_operation

        if name_operation != 'None':
            if DICT_OPERS_SUFFIX[name_operation]:
                suffix_output_names += '_'+ DICT_OPERS_SUFFIX[name_operation]
    #endfor

    if args.out_nifti:
        in_file_extension = '.nii'
    else:
        in_file_extension = fileextension(listInputFiles[0])

    if args.no_suffix_outname:
        nameOutputFiles = lambda in_name: basename_file_noext(in_name) + in_file_extension
    else:
        nameOutputFiles = lambda in_name: basename_file_noext(in_name) + suffix_output_names + in_file_extension



    for i, in_file in enumerate(listInputFiles):
        print("\nInput: \'%s\'..." % (basename(in_file)))

        inout_data = ImageFileReader.get_image(in_file)

        if is_update_metadata_file:
            inout_metadata = func_updatemetadata(in_file, i)
        else:
            inout_metadata = ImageFileReader.get_image_metadata_info(in_file)


        for func_name, func_operation in dict_func_operations.items():
            inout_data = func_operation(inout_data, i)  # perform whichever operation
        #endfor


        out_filename = join_path_names(args.output_dir, nameOutputFiles(basename(in_file)))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_filename), str(inout_data.shape)))

        ImageFileReader.write_image(out_filename, inout_data, metadata=inout_metadata)
    #endfor



if __name__ == "__main__":
    dict_opers_help = {'mask': 'apply mask to image',
                       'binarise': 'binarise a mask with multiple labels',
                       'merge': 'merge (add) two masks',
                       'substract': 'substract the 2nd mask to the 1st mask',
                       'crop': 'crop image to bounding box (input arg)',
                       'rescale': 'rescale image to factor (input arg)',
                       'rescalemask': 'rescale mask to factor (input arg), keep output format (0, 1)',
                       'fillholes': 'apply a morphological fill of holes inside a mask',
                       'erode': 'apply a morphological erosion to mask (1 layer voxels)',
                       'dilate': 'apply a morphological dilation to mask (1 layer voxels)',
                       'moropen': 'apply a morphological opening to mask (1 layer voxels)',
                       'morclose': 'apply a morphological closing to mask (1 layer voxels)',
                       'connregs': 'split mask in connected components. Get each component with different label',
                       'firstconreg': 'split mask in connected components and get the one with largest volume',
                       'thinning': 'apply morphological thinning to mask and obtain centrelines',
                       'threshold': 'apply threshold of probability image and obtain binary mask',
                       'masklabels': 'retrieve the labels (input arg) and get mask containing these',
                       'normalise': 'normalise images to have voxels in the range (0, 1)',
                       'power': 'compute power of mask values',
                       'exponential': 'compute exponential of mask values'}
    string_opers_help = '\n'.join([(key+': '+val) for key,val in dict_opers_help.items()])
    
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--type', nargs='+', default=['None'], help=string_opers_help)
    parser.add_argument('--input_roidir', type=str, default=None)
    parser.add_argument('--input_2nddir', type=str, default=None)
    parser.add_argument('--reference_dir', type=str, default=None)
    parser.add_argument('--boundbox_file', type=str, default=None)
    parser.add_argument('--rescale_file', type=str, default=None)
    parser.add_argument('--in_mask_labels', nargs='+', type=int, default=None)
    parser.add_argument('--out_nifti', type=str2bool, default=False)
    parser.add_argument('--no_suffix_outname', type=str2bool, default=None)
    args = parser.parse_args()

    for ioperation in args.type:
        if ioperation not in LIST_OPERATIONS:
            message = 'Type operation chosen \'%s\' not available' %(ioperation)
            catch_error_exception(message)

    if 'mask' in args.type:
        if not args.input_roidir:
            message = 'need to set argument \'input_roidir\''
            catch_error_exception(message)
    if 'merge' in args.type or 'substract' in args.type:
        if not args.input_2nddir:
            message = 'need to set argument \'input_2nddir\''
            catch_error_exception(message)
    elif 'crop' in args.type:
        if not args.reference_dir or not args.boundbox_file:
            message = 'need to set arguments \'reference_dir\' and \'boundbox_file\''
            catch_error_exception(message)
    elif 'rescale' in args.type or \
        'rescale_mask' in args.type:
        if not args.reference_dir or not args.rescale_file:
            message = 'need to set arguments \'reference_dir\' and \'rescale_file\''
            catch_error_exception(message)
    elif 'masklabels' in args.type:
        if not args.in_mask_labels:
            message = 'need to set arguments \'in_mask_labels\''
            catch_error_exception(message)

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)