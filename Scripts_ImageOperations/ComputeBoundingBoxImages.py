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
from Preprocessing.BoundingBoxes import *
from Preprocessing.OperationImages import *
from collections import OrderedDict
import argparse



def main(args):
    # ---------- SETTINGS ----------
    max_size_bounding_box = (0, 0, 0)
    min_size_bounding_box = (1.0e+03, 1.0e+03, 1.0e+03)
    voxels_buffer_border = (0, 0, 0, 0)

    nameInputRelPath = 'Images_Rescaled_Full'
    nameRoiMaskRelPath = 'Lungs_Rescaled_Full'

    nameInputFiles = '*.nii.gz'
    nameRoiMasksFiles = '*.nii.gz'

    nameCropBoundingBoxes_FileNpy = 'cropBoundingBoxes_352x240_rescaled0.6x0.6x0.6_images.npy'
    nameCropBoundingBoxes_FileCsv = 'cropBoundingBoxes_352x240_rescaled0.6x0.6x0.6_images.csv'
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(args.basedir)
    BaseDataPath = workDirsManager.getNameBaseDataPath()
    InputPath = workDirsManager.getNameExistPath(BaseDataPath, nameInputRelPath)
    RoiMasksPath = workDirsManager.getNameExistPath(BaseDataPath, nameRoiMaskRelPath)

    listInputImageFiles = findFilesDirAndCheck(InputPath, nameInputFiles)
    listRoiMaskFiles = findFilesDirAndCheck(RoiMasksPath, nameRoiMasksFiles)


    dict_cropBoundingBoxes = OrderedDict()

    for in_file, roimask_file in zip(listInputImageFiles, listRoiMaskFiles):
        print("\nInput: \'%s\'..." % (basename(in_file)))
        print("RoI mask (lungs) file: \'%s\'..." % (basename(in_file)))

        roimask_array = FileReader.getImageArray(roimask_file)

        bounding_box = BoundingBoxes.compute_bounding_box_contain_masks_with_border_effects_2D(roimask_array,
                                                                                               voxels_buffer_border= voxels_buffer_border)

        size_bounding_box = BoundingBoxes.compute_size_bounding_box(bounding_box)
        print("Bounding-box: \'%s\', of size: \'%s\'" %(bounding_box, size_bounding_box))

        max_size_bounding_box = BoundingBoxes.compute_max_size_bounding_box(size_bounding_box,
                                                                            max_size_bounding_box)

        # Compute new bounding-box, of fixed size 'args.cropSizeBoundingBox', and with same center as original 'bounding_box'
        processed_bounding_box = BoundingBoxes.compute_bounding_box_centered_bounding_box_2D(bounding_box,
                                                                                             args.cropSizeBoundingBox,
                                                                                             roimask_array.shape)

        size_processed_bounding_box = BoundingBoxes.compute_size_bounding_box(processed_bounding_box)
        print("Processed bounding-box: \'%s\', of size: \'%s\'" %(processed_bounding_box, size_processed_bounding_box))

        dict_cropBoundingBoxes[filenamenoextension(in_file)] = processed_bounding_box
    #endfor

    print("max size bounding-box found: \'%s\'; set size bounding-box \'%s\'" %(max_size_bounding_box, args.cropSizeBoundingBox))


    # Save dictionary in file
    nameoutfile = joinpathnames(BaseDataPath, nameCropBoundingBoxes_FileNpy)
    saveDictionary(nameoutfile, dict_cropBoundingBoxes)
    nameoutfile = joinpathnames(BaseDataPath, nameCropBoundingBoxes_FileCsv)
    saveDictionary_csv(nameoutfile, dict_cropBoundingBoxes)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--cropSizeBoundingBox', type=str2tuplefloat, default=CROPSIZEBOUNDINGBOX)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
