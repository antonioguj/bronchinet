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



def main(args):
    # ---------- SETTINGS ----------
    nameInputRoiMaskRelPath       = 'Lungs/'
    nameReferenceFilesRelPath     = 'RawImages/'
    nameOutputDataRelPath         = 'CroppingData/'
    nameCropBoundingBoxes_FileNpy = 'cropBoundingBoxes_images_fullsize.npy'
    nameCropBoundingBoxes_FileCsv = 'cropBoundingBoxes_images_fullsize.csv'
    size_borders_buffer = (20, 20, 20)
    # ---------- SETTINGS ----------


    workDirsManager    = WorkDirsManager(args.datadir)
    InputRoiMasksPath  = workDirsManager.getNameExistPath(nameInputRoiMaskRelPath)
    ReferenceFilesPath = workDirsManager.getNameExistPath(nameReferenceFilesRelPath)
    OutputDataPath     = workDirsManager.getNameNewPath(nameOutputDataRelPath)

    listInputRoiMaskFiles = findFilesDirAndCheck(InputRoiMasksPath)
    listReferenceFiles    = findFilesDirAndCheck(ReferenceFilesPath)



    dict_cropBoundingBoxes = OrderedDict()
    max_size_bounding_box = (0, 0, 0)
    min_size_bounding_box = (1.0e+03, 1.0e+03, 1.0e+03)

    for i, in_roimask_file in enumerate(listInputRoiMaskFiles):
        print("\nInput: \'%s\'..." % (basename(in_roimask_file)))

        in_roimask_array = FileReader.getImageArray(in_roimask_file)
        print("Dims : \'%s\'..." % (str(in_roimask_array.shape)))

        # convert train masks to binary (0, 1)
        in_roimask_array = OperationBinaryMasks.process_masks(in_roimask_array)


        bounding_box = BoundingBoxes.compute_bounding_box_contain_masks(in_roimask_array,
                                                                        size_borders_buffer=size_borders_buffer,
                                                                        is_bounding_box_slices=args.isCalcBoundingBoxInSlices)
        size_bounding_box = BoundingBoxes.get_size_bounding_box(bounding_box)
        print("Bounding box: \'%s\', of size: \'%s\'" %(bounding_box, size_bounding_box))

        max_size_bounding_box = BoundingBoxes.get_max_size_bounding_box(size_bounding_box, max_size_bounding_box)
        min_size_bounding_box = BoundingBoxes.get_min_size_bounding_box(size_bounding_box, min_size_bounding_box)


        # store computed bounding-box
        in_reference_file = listReferenceFiles[i]
        dict_cropBoundingBoxes[filenamenoextension(in_reference_file)] = bounding_box
    #endfor

    print("\nMax size bounding box found: \'%s\'..." %(str(max_size_bounding_box)))
    print("Min size bounding box found: \'%s\'..." %(str(min_size_bounding_box)))



    if args.isSameBoundBoxSizeAllImages:
        print("Compute bounding boxes of same size for all images...")

        if args.fixedSizeBoundingBox:
            final_size_bounding_box = args.fixedSizeBoundingBox
        else:
            # if not fixed scale specified, take the maximum bounding box computed
            final_size_bounding_box = max_size_bounding_box
        print("Final size bounding box: \'%s\'..." %(str(final_size_bounding_box)))


        dict_cropBoundingBoxes_prev = dict_cropBoundingBoxes
        dict_cropBoundingBoxes = OrderedDict()

        for i, in_roimask_file in enumerate(listInputRoiMaskFiles):
            print("\nInput: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask_array_shape = FileReader.getImageSize(in_roimask_file)
            print("Dims : \'%s\'..." % (str(in_roimask_array_shape)))

            in_reference_file = listReferenceFiles[i]
            in_bounding_box = dict_cropBoundingBoxes_prev[filenamenoextension(in_reference_file)]


            # Compute new bounding-box, of fixed size 'args.cropSizeBoundingBox', and with same center as original 'bounding_box'
            out_bounding_box = BoundingBoxes.compute_bounding_box_centered_bounding_box_fit_image(in_bounding_box,
                                                                                                  final_size_bounding_box,
                                                                                                  in_roimask_array_shape,
                                                                                                  is_bounding_box_slices=args.isCalcBoundingBoxInSlices)
            size_bounding_box = BoundingBoxes.get_size_bounding_box(out_bounding_box)
            print("Processed bounding box: \'%s\', of size: \'%s\'" % (out_bounding_box, size_bounding_box))


            # store computed bounding-box
            in_reference_file = listReferenceFiles[i]
            dict_cropBoundingBoxes[filenamenoextension(in_reference_file)] = out_bounding_box
        #endfor
    #endfor


    # Save dictionary in file
    nameoutfile = joinpathnames(OutputDataPath, nameCropBoundingBoxes_FileNpy)
    saveDictionary(nameoutfile, dict_cropBoundingBoxes)
    nameoutfile = joinpathnames(OutputDataPath, nameCropBoundingBoxes_FileCsv)
    saveDictionary_csv(nameoutfile, dict_cropBoundingBoxes)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--isSameBoundBoxSizeAllImages', type=str2bool, default=ISSAMEBOUNDBOXSIZEALLIMAGES)
    parser.add_argument('--fixedSizeBoundingBox', type=str2tuplefloat, default=FIXEDSIZEBOUNDINGBOX)
    parser.add_argument('--isCalcBoundingBoxInSlices', type=str2bool, default=ISCALCBOUNDINGBOXINSLICES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
