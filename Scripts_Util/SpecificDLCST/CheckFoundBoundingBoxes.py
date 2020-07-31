#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from common.exception_manager import *
from dataloaders.imagefilereader import *
from imageoperators.imageoperator import *
import argparse



def main(args):

    listInputCropImagesFiles = list_files_dir(args.cropimagesdir)
    listInputFullImagesFiles = list_files_dir(args.fullimagesdir)
    dictInputBoundingBoxes   = read_dictionary(args.inputBoundBoxesFile)


    names_files_different = []

    for i, (in_cropimage_file, in_fullimage_file) in enumerate(zip(listInputCropImagesFiles, listInputFullImagesFiles)):
        print("\nInput: \'%s\'..." % (basename(in_cropimage_file)))
        print("Input 2: \'%s\'..." % (basename(in_fullimage_file)))

        in_cropimage_array = ImageFileReader.get_image(in_cropimage_file)
        in_fullimage_array = ImageFileReader.get_image(in_fullimage_file)


        # 1 step: crop image
        in_bounding_box     = dictInputBoundingBoxes[basename_file_noext(in_cropimage_file)]
        new_cropimage_array = CropImage._compute3D(in_fullimage_array, in_bounding_box)
        # 2 step: invert image
        new_cropimage_array = FlipImage.compute(new_cropimage_array, axis=0)

        if (in_cropimage_array.shape == new_cropimage_array.shape):
            res_voxels_equal = np.array_equal(in_cropimage_array, new_cropimage_array)

            if res_voxels_equal:
                print("GOOD: Images are equal voxelwise...")
            else:
                names_files_different.append(basename(in_cropimage_file))
                message = "ERROR: Images are different..."
                catch_warning_exception(message)
        else:
            names_files_different.append(basename(in_cropimage_file))
            message = "ERROR: Images are different..."
            catch_warning_exception(message)
    #endfor

    if (len(names_files_different) == 0):
        print("\nGOOD: ALL IMAGE FILES ARE EQUAL...")
    else:
        print("\nERROR: Found \'%s\' files that are different. Names of files: \'%s\'..." %(len(names_files_different),
                                                                                            names_files_different))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cropimagesdir', type=str)
    parser.add_argument('fullimagesdir', type=str)
    parser.add_argument('--inputBoundBoxesFile', type=str, default='found_boundingBox_croppedCTinFull.npy')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)
