#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from common.constant import *
from common.workdirmanager import *
from dataloaders.imagefilereader import *
from imageoperators.boundingboxes import *
from imageoperators.imageoperator import *
from imageoperators.maskoperator import *
from collections import OrderedDict
import argparse



def main(args):

    workDirsManager     = GeneralDirManager(args.datadir)
    InputRoiMasksPath   = workDirsManager.get_pathdir_exist(args.nameInputRoiMasksRelPath)
    InputReferKeysPath  = workDirsManager.get_pathdir_exist(args.nameInputReferKeysRelPath)
    OutputBoundingBoxesFile= workDirsManager.get_pathfile_update(args.nameOutputBoundingBoxesFile)

    listInputRoiMasksFiles  = list_files_dir(InputRoiMasksPath)
    listInputReferKeysFiles = list_files_dir(InputReferKeysPath)



    outdict_cropBoundingBoxes = OrderedDict()
    max_size_bounding_box = (0, 0, 0)
    min_size_bounding_box = (1.0e+03, 1.0e+03, 1.0e+03)

    if args.isTwoBoundingBoxEachLungs:
        # compute two bounding-boxes separately, for both left and right lungs
        for i, in_roimask_file in enumerate(listInputRoiMasksFiles):
            print("\nInput: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask_array = ImageFileReader.get_image(in_roimask_file)
            print("Dims : \'%s\'..." % (str(in_roimask_array.shape)))

            # extract masks corresponding to left (=1) and right (=2) lungs
            (in_roimask_left_array, in_roimask_right_array)  = MaskOperator.get_list_masks_with_labels(in_roimask_array, [1, 2])


            # left lung bounding-box:
            bounding_box_left = BoundingBoxes.compute_bounding_box_contain_masks(in_roimask_left_array,
                                                                                 size_borders_buffer=args.sizeBufferInBorders,
                                                                                 is_bounding_box_slices=args.isCalcBoundingBoxInSlices)
            size_bounding_box = BoundingBoxes.get_size_bounding_box(bounding_box_left)
            print("Bounding-box around ROI: left lung: \'%s\', of size: \'%s\'" % (bounding_box_left, size_bounding_box))

            max_size_bounding_box = BoundingBoxes.get_max_size_bounding_box(size_bounding_box, max_size_bounding_box)
            min_size_bounding_box = BoundingBoxes.get_min_size_bounding_box(size_bounding_box, min_size_bounding_box)

            # right lung bounding-box:
            bounding_box_right = BoundingBoxes.compute_bounding_box_contain_masks(in_roimask_right_array,
                                                                                 size_borders_buffer=args.sizeBufferInBorders,
                                                                                 is_bounding_box_slices=args.isCalcBoundingBoxInSlices)
            size_bounding_box = BoundingBoxes.get_size_bounding_box(bounding_box_right)
            print("Bounding-box around ROI: right lung: \'%s\', of size: \'%s\'" % (bounding_box_right, size_bounding_box))


            # store computed bounding-boxes
            in_referkey_file = listInputReferKeysFiles[i]
            outdict_cropBoundingBoxes[basename_file_noext(in_referkey_file)] = [bounding_box_left, bounding_box_right]
        #endfor
    else:
        # compute bounding-box including for both left and right lungs
        for i, in_roimask_file in enumerate(listInputRoiMasksFiles):
            print("\nInput: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask_array = ImageFileReader.get_image(in_roimask_file)
            print("Dims : \'%s\'..." % (str(in_roimask_array.shape)))

            # convert train masks to binary (0, 1)
            in_roimask_array = MaskOperator.binarise(in_roimask_array)


            bounding_box = BoundingBoxes.compute_bounding_box_contain_masks(in_roimask_array,
                                                                            size_borders_buffer=args.sizeBufferInBorders,
                                                                            is_bounding_box_slices=args.isCalcBoundingBoxInSlices)
            size_bounding_box = BoundingBoxes.get_size_bounding_box(bounding_box)
            print("Bounding-box around ROI: lungs: \'%s\', of size: \'%s\'" % (bounding_box, size_bounding_box))

            max_size_bounding_box = BoundingBoxes.get_max_size_bounding_box(size_bounding_box, max_size_bounding_box)
            min_size_bounding_box = BoundingBoxes.get_min_size_bounding_box(size_bounding_box, min_size_bounding_box)


            # store computed bounding-box
            in_referkey_file = listInputReferKeysFiles[i]
            outdict_cropBoundingBoxes[basename_file_noext(in_referkey_file)] = bounding_box
    #endfor

    print("\nMax size bounding-box found: \'%s\'..." %(str(max_size_bounding_box)))
    print("Min size bounding-box found: \'%s\'..." %(str(min_size_bounding_box)))



    print("\nPost-process the Computed Bounding-Boxes:")
    if args.isSameSizeBoundBoxAllImages:
        print("Compute bounding-boxes of same size for all images...")

        if args.fixedSizeBoundingBox:
            fixed_size_bounding_box = args.fixedSizeBoundingBox
            print("Fixed size bounding-box from input: \'%s\'..." %(str(fixed_size_bounding_box)))
        else:
            fixed_size_bounding_box = max_size_bounding_box
            print("Fixed size bounding-box as the max. size of computed bounding-boxes: \'%s\'..." %(str(fixed_size_bounding_box)))
    else:
        fixed_size_bounding_box = False


    for i, (in_key_file, in_list_bounding_boxes) in enumerate(outdict_cropBoundingBoxes.items()):
        print("\nInput Key file: \'%s\'..." % (in_key_file))

        in_roimask_file = listInputRoiMasksFiles[i]
        in_roimask_array_shape = ImageFileReader.get_image_size(in_roimask_file)
        print("Assigned to file: \'%s\', of Dims : \'%s\'..." % (basename(in_roimask_file), str(in_roimask_array_shape)))

        if not args.isTwoBoundingBoxEachLungs:
            in_list_bounding_boxes = [in_list_bounding_boxes]


        for j, in_bounding_box in enumerate(in_list_bounding_boxes):
            size_bounding_box = BoundingBoxes.get_size_bounding_box(in_bounding_box)
            print("\nInput Bounding-box: \'%s\', of size: \'%s\'" % (in_bounding_box, size_bounding_box))


            if args.isSameSizeBoundBoxAllImages:
                if args.isCalcBoundingBoxInSlices:
                    size_proc_bounding_box = (size_bounding_box[0],) + fixed_size_bounding_box
                else:
                    size_proc_bounding_box = fixed_size_bounding_box
                print("Resize bounding-box to fixed final size: \'%s\'..." %(str(size_proc_bounding_box)))
            else:
                size_proc_bounding_box = size_bounding_box

            # Check whether the size of bounding-box is smaller than the training image patches
            if not BoundingBoxes.is_image_patch_contained_in_bounding_box(size_proc_bounding_box, args.sizeTrainImages):
                print("Size of bounding-box is smaller than size of training image patches: \'%s\'..." %(str(args.sizeTrainImages)))

                size_proc_bounding_box = BoundingBoxes.get_max_size_bounding_box(size_proc_bounding_box, args.sizeTrainImages)
                print("Resize bounding-box to size: \'%s\'..." %(str(size_proc_bounding_box)))
            #endif


            if size_proc_bounding_box != size_bounding_box:
                # Compute new bounding-box: of 'size_proc_bounding_box', with same center as 'in_bounding_box', and that fits in 'in_roimask_array'
                proc_bounding_box = BoundingBoxes.compute_bounding_box_centered_bounding_box_fit_image(in_bounding_box,
                                                                                                       size_proc_bounding_box,
                                                                                                       in_roimask_array_shape)
                size_proc_bounding_box = BoundingBoxes.get_size_bounding_box(proc_bounding_box)
                print("New processed bounding-box: \'%s\', of size: \'%s\'" % (proc_bounding_box, size_proc_bounding_box))

                # store computed bounding-box
                if args.isTwoBoundingBoxEachLungs:
                    outdict_cropBoundingBoxes[in_key_file][j] = proc_bounding_box
                else:
                    outdict_cropBoundingBoxes[in_key_file] = proc_bounding_box
            else:
                print("No changes made to bounding-box...")
        #endfor
    #endfor


    # Save dictionary in file
    save_dictionary(OutputBoundingBoxesFile, outdict_cropBoundingBoxes)
    save_dictionary_csv(OutputBoundingBoxesFile.replace('.npy', '.csv'), outdict_cropBoundingBoxes)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--nameInputRoiMasksRelPath', type=str, default=NAME_RAWROIMASKS_RELPATH)
    parser.add_argument('--nameInputReferKeysRelPath', type=str, default=NAME_REFERKEYS_RELPATH)
    parser.add_argument('--nameOutputBoundingBoxesFile', type=str, default=NAME_CROPBOUNDINGBOX_FILE)
    parser.add_argument('--isTwoBoundingBoxEachLungs', type=str2bool, default=ISTWOBOUNDINGBOXEACHLUNGS)
    parser.add_argument('--sizeBufferInBorders', type=str2tuple_int, default=SIZEBUFFERBOUNDBOXBORDERS)
    parser.add_argument('--sizeTrainImages', type=str2tuple_int, default=IMAGES_DIMS_Z_X_Y)
    parser.add_argument('--isSameSizeBoundBoxAllImages', type=str2bool, default=ISSAMESIZEBOUNDBOXALLIMAGES)
    parser.add_argument('--fixedSizeBoundingBox', type=str2tuple_int, default=FIXEDSIZEBOUNDINGBOX)
    parser.add_argument('--isCalcBoundingBoxInSlices', type=str2bool, default=ISCALCBOUNDINGBOXINSLICES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)
