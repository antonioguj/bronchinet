#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.Constants import *
from CommonUtil.ErrorMessages import *
from CommonUtil.FileReaders import *
from CommonUtil.FunctionsUtil import *
from CommonUtil.WorkDirsManager import *
import argparse



def main(args):

    # ---------- SETTINGS ----------
    namesFilesDir1 = '*_lumen_airway-trachea.nii.gz'
    namesFilesDir2 = '*_lumen_airway-trachea.nii.gz'

    max_rel_error_images = 1.0e-06
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(args.basedir)
    Directory1Path  = workDirsManager.getNameExistPath(args.basedir, args.dir1)
    Directory2Path  = workDirsManager.getNameExistPath(args.basedir, args.dir2)

    listImagesFilesDir1 = findFilesDir(Directory1Path, namesFilesDir1)
    listImagesFilesDir2 = findFilesDir(Directory2Path, namesFilesDir2)

    nbImagesFilesDir1 = len(listImagesFilesDir1)
    nbImagesFilesDir2 = len(listImagesFilesDir2)

    # Run checkers
    if (nbImagesFilesDir1 == 0):
        message = "0 Images found in dir \'%s\'" %(Directory1Path)
        CatchErrorException(message)
    if (nbImagesFilesDir1 != nbImagesFilesDir2):
        message = "num Images in dir 1: %i, not equal to num Images in dir 2: %i" %(nbImagesFilesDir1, nbImagesFilesDir2)
        CatchErrorException(message)



    names_files_different = []

    for images_1_file, images_2_file in zip(listImagesFilesDir1, listImagesFilesDir2):

        print('file 1: \'%s\'...' % (images_1_file))
        print('file 2: \'%s\'...' % (images_2_file))

        images_1_array = FileReader.getImageArray(images_1_file)
        images_2_array = FileReader.getImageArray(images_2_file)

        if (images_1_array.shape != images_2_array.shape):
            message = "size of images 1 and 2 not equal: %s != %s" %(images_1_array.shape, images_2_array.shape)
            CatchWarningException(message)

            names_files_different.append(basename(images_1_file))
            continue


        result = np.array_equal(images_1_array, images_2_array)

        if (result==False):
            message = "Images 1 and 2 are not equal"
            CatchWarningException(message)

            error_images_array = images_1_array - images_2_array

            num_voxels_diff      = np.count_nonzero(error_images_array)
            num_voxels_1_nonzero = np.count_nonzero(images_1_array)
            rel_num_voxels_diff  = (num_voxels_diff / float(num_voxels_1_nonzero)) * 100

            mean_error_images_array = abs(np.mean(error_images_array))
            mean_images_1_array     = abs(np.mean(images_1_array))
            rel_error_images_array  = (mean_error_images_array / float(mean_images_1_array)) * 100

            print("Found num. voxels different: %s. Rel percentage: %s. Error between 2 images: %s..." %(num_voxels_diff,
                                                                                                         rel_num_voxels_diff,
                                                                                                         rel_error_images_array))
            if rel_error_images_array < max_rel_error_images:
                print("GOOD. Error lower than tolerance. Images are equal...")
            else:
                message = "Error larger than tolerance. Images are different"
                CatchWarningException(message)

                names_files_different.append(basename(images_2_file))
                continue
        else:
            print("GOOD: Images are equal...")
    #endfor


    if (len(names_files_different)==0):
        print("Good: All images files are equal...")
    else:
        print("Found %s files that are different..." %(len(names_files_different)))
        print("Names of files: %s..." % (names_files_different))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--dir1')
    parser.add_argument('--dir2')
    args = parser.parse_args()

    if not args.dir1:
        message = "Please input valid directory 1"
        CatchErrorException(message)
    if not args.dir2:
        message = "Please input valid directory 2"
        CatchErrorException(message)

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)