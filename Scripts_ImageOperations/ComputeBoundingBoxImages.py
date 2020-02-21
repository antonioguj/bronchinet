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
from OperationImages.BoundingBoxes import *
from OperationImages.OperationImages import *
from OperationImages.OperationMasks import *
from collections import OrderedDict
import argparse


def isSizeBoundingBoxSmallerThanSizeImages(size_bounding_box, size_images):
    return any((el_a < el_b) for (el_a, el_b) in zip(size_bounding_box, size_images))



def main(args):
    # ---------- SETTINGS ----------
    size_borders_buffer = (20, 20, 20)
    nameCropBoundingBoxesFile = 'cropBoundingBoxes_images.npy'

    # def funcNameCropBoundingBoxesFile(fix_boundBox):
    #     if args.isSameBoundBoxSizeAllImages:
    #         suffix = '-'.join(['%0.3i'%(elem) for elem in fix_boundBox])
    #     else:
    #         suffix = 'fitted'
    #     return 'cropBoundingBoxes_images_%s.npy' %(suffix)
    # ---------- SETTINGS ----------


    workDirsManager   = WorkDirsManager(args.datadir)
    InputRoiMasksPath = workDirsManager.getNameExistPath(args.nameInputRoiMasksRelPath)
    InReferKeysPath   = workDirsManager.getNameExistPath(args.nameInReferKeysRelPath)

    listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath)
    listInReferKeysFiles   = findFilesDirAndCheck(InReferKeysPath)



    dict_cropBoundingBoxes = OrderedDict()
    max_size_bounding_box = (0, 0, 0)
    min_size_bounding_box = (1.0e+03, 1.0e+03, 1.0e+03)

    for i, in_roimask_file in enumerate(listInputRoiMasksFiles):
        print("\nInput: \'%s\'..." % (basename(in_roimask_file)))

        in_roimask_array = FileReader.getImageArray(in_roimask_file)
        print("Dims : \'%s\'..." % (str(in_roimask_array.shape)))

        # convert train masks to binary (0, 1)
        in_roimask_array = OperationBinaryMasks.process_masks(in_roimask_array)


        bounding_box = BoundingBoxes.compute_bounding_box_contain_masks(in_roimask_array,
                                                                        size_borders_buffer=size_borders_buffer,
                                                                        is_bounding_box_slices=args.isCalcBoundingBoxInSlices)
        size_bounding_box = BoundingBoxes.get_size_bounding_box(bounding_box)
        print("Bounding-box: \'%s\', of size: \'%s\'" %(bounding_box, size_bounding_box))


        # Check whether the size of bounding-box is smaller tha training image patches
        if isSizeBoundingBoxSmallerThanSizeImages(size_bounding_box, args.sizeInputTrainImages):
            print("WARNING: Size of bounding-box: \'%s\' smaller than size of input image patches: \'%s\'..." %(size_bounding_box,
                                                                                                                args.sizeInputTrainImages))

            new_size_bounding_box = tuple([max(el_a, el_b) for (el_a, el_b) in zip(size_bounding_box,
                                                                                   args.sizeInputTrainImages)])

            # Compute new bounding-box, of fixed size 'new_size_bounding_box', and with same center as original 'bounding_box'
            bounding_box = BoundingBoxes.compute_bounding_box_centered_bounding_box_fit_image(bounding_box,
                                                                                              new_size_bounding_box,
                                                                                              in_roimask_array.shape)
            size_bounding_box = BoundingBoxes.get_size_bounding_box(bounding_box)
            print("Computed new bounding-box: \'%s\', of size: \'%s\'" % (bounding_box, size_bounding_box))
        #endif


        max_size_bounding_box = BoundingBoxes.get_max_size_bounding_box(size_bounding_box, max_size_bounding_box)
        min_size_bounding_box = BoundingBoxes.get_min_size_bounding_box(size_bounding_box, min_size_bounding_box)


        # store computed bounding-box
        in_referkey_file = listInReferKeysFiles[i]
        dict_cropBoundingBoxes[filenamenoextension(in_referkey_file)] = bounding_box
    #endfor

    print("\nMax size bounding-box found: \'%s\'..." %(str(max_size_bounding_box)))
    print("Min size bounding-box found: \'%s\'..." %(str(min_size_bounding_box)))



    final_size_bounding_box = 0
    if args.isSameBoundBoxSizeAllImages:
        print("Compute bounding-boxes of same size for all images...")

        if args.fixedSizeBoundingBox:
            final_size_bounding_box = args.fixedSizeBoundingBox
        else:
            # if not fixed scale specified, take the maximum bounding box computed
            final_size_bounding_box = max_size_bounding_box
        print("Final size bounding-box: \'%s\'..." %(str(final_size_bounding_box)))


        dict_cropBoundingBoxes_prev = dict_cropBoundingBoxes
        dict_cropBoundingBoxes = OrderedDict()

        for i, in_roimask_file in enumerate(listInputRoiMasksFiles):
            print("\nInput: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask_array_shape = FileReader.getImageSize(in_roimask_file)
            print("Dims : \'%s\'..." % (str(in_roimask_array_shape)))

            in_referkey_file = listInReferKeysFiles[i]
            in_bounding_box = dict_cropBoundingBoxes_prev[filenamenoextension(in_referkey_file)]


            # Compute new bounding-box, of fixed size 'args.cropSizeBoundingBox', and with same center as original 'bounding_box'
            out_bounding_box = BoundingBoxes.compute_bounding_box_centered_bounding_box_fit_image(in_bounding_box,
                                                                                                  final_size_bounding_box,
                                                                                                  in_roimask_array_shape,
                                                                                                  is_bounding_box_slices=args.isCalcBoundingBoxInSlices)
            size_bounding_box = BoundingBoxes.get_size_bounding_box(out_bounding_box)
            print("Processed bounding-box: \'%s\', of size: \'%s\'" % (out_bounding_box, size_bounding_box))


            # store computed bounding-box
            in_referkey_file = listInReferKeysFiles[i]
            dict_cropBoundingBoxes[filenamenoextension(in_referkey_file)] = out_bounding_box
        #endfor
    #endif


    # Save dictionary in file
    # nameCropBoundingBoxesFile = funcNameCropBoundingBoxesFile(final_size_bounding_box)
    if isExistfile(nameCropBoundingBoxesFile):
        nameCropBoundingBoxesFile = nameCropBoundingBoxesFile.replace('.npy', '_next.npy')

    nameoutfile = joinpathnames(args.datadir, nameCropBoundingBoxesFile)
    saveDictionary(nameoutfile, dict_cropBoundingBoxes)
    nameoutfile = joinpathnames(args.datadir, nameCropBoundingBoxesFile.replace('.npy','.csv'))
    saveDictionary_csv(nameoutfile, dict_cropBoundingBoxes)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--nameInputRoiMasksRelPath', type=str, default=NAME_RAWROIMASKS_RELPATH)
    parser.add_argument('--nameInReferKeysRelPath', type=str, default=NAME_REFERKEYS_RELPATH)
    parser.add_argument('--rescaleImages', type=str2bool, default=RESCALEIMAGES)
    parser.add_argument('--nameRescaleFactorFile', type=str, default=NAME_RESCALEFACTOR_FILE)
    parser.add_argument('--sizeInputTrainImages', type=str2tupleint, default=IMAGES_DIMS_Z_X_Y)
    parser.add_argument('--isSameBoundBoxSizeAllImages', type=str2bool, default=ISSAMEBOUNDBOXSIZEALLIMAGES)
    parser.add_argument('--fixedSizeBoundingBox', type=str2tuplefloat, default=FIXEDSIZEBOUNDINGBOX)
    parser.add_argument('--isCalcBoundingBoxInSlices', type=str2bool, default=ISCALCBOUNDINGBOXINSLICES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
