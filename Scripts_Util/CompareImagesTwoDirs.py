#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

import argparse

from DataLoaders.FileReaders import *
from OperationImages.OperationImages import MorphoOpenImages
from PlotsManager.Histograms import *

_max_relerror = 1.0e-06



def main(args):
    # ---------- SETTINGS ----------
    InputPath1   = args.inputdir1
    InputPath2   = args.inputdir2
    nameOutDiffImageFilesName = 'out_absdiffimgs_image%s.nii.gz'
    nameOutHistoFilesName = 'out_histogram_image%s.png'
    # ---------- SETTINGS ----------


    listInputFiles1 = findFilesDirAndCheck(InputPath1)
    listInputFiles2 = findFilesDirAndCheck(InputPath2)

    if (len(listInputFiles1) != len(listInputFiles2)):
       message = 'num files in dir 1 \'%s\', not equal to num files in dir 2 \'%i\'...' %(len(listInputFiles1), len(listInputFiles2))
       CatchErrorException(message)

    if not isExistdir(args.tempdir):
        makedir(args.tempdir)



    names_files_different = []

    for i, (in_file_1, in_file_2) in enumerate(zip(listInputFiles1, listInputFiles2)):
        print("\nInput: \'%s\'..." % (in_file_1))
        print("And: \'%s\'..." % (in_file_2))

        in_image1_array = FileReader.getImageArray(in_file_1)
        in_image2_array = FileReader.getImageArray(in_file_2)
        #print("Values in input array 1: %s..." % (np.unique(in_image1_array)))
        #print("Values in input array 2: %s..." % (np.unique(in_image2_array)))


        if (in_image1_array.shape == in_image2_array.shape):
            is_arrays_equal_voxelwise = np.array_equal(in_image1_array, in_image2_array)

            if is_arrays_equal_voxelwise:
                print("GOOD: Images are equal voxelwise...")
            else:
                print("WARNING: Images 1 and 2 are not equal...")
                print("Do morphological opening of difference between images to remove strayed voxels due to noise...")

                out_diffimages_array = abs(in_image1_array - in_image2_array)

                out_diffimages_array = MorphoOpenImages.compute(out_diffimages_array)

                num_voxels_diffimages = np.count_nonzero(out_diffimages_array)

                if num_voxels_diffimages==0:
                    print("GOOD: After a morphological opening, Images are equal...")
                else:
                    print("WARNING: After a morphological opening, Images 1 and 2 are still not equal...")
                    print("Analyse global error magnitudes of the difference between images...")

                    num_voxels_nonzero_image1 = np.count_nonzero(in_image1_array)
                    relerror_voxels_diffimages = (num_voxels_diffimages / float(num_voxels_nonzero_image1))

                    print("Num voxels of difference between images \'%s\' out of total non-zero voxels in image 1 \'%s\'. Relative error \'%s\'..."
                          %(num_voxels_diffimages, num_voxels_nonzero_image1, relerror_voxels_diffimages))

                    absmean_intens_diffimages  = abs(np.mean(out_diffimages_array))
                    absmean_intens_image1      = abs(np.mean(in_image1_array))
                    relerror_intens_diffimages = (absmean_intens_diffimages / absmean_intens_image1)
                    print("Mean value of Intensity of difference between images \'%s\'. Relative error \'%s\'..."
                          %(absmean_intens_diffimages, relerror_intens_diffimages))


                    if relerror_intens_diffimages < _max_relerror:
                        print("GOOD. Relative error \'%s\' lower than tolerance \'%s\'. Images can be considered equal..."
                              % (relerror_intens_diffimages, _max_relerror))
                    else:
                        print("WARNING. Relative error \'%s\' larger than tolerance \'%s\'. Images are different..."
                              % (relerror_intens_diffimages, _max_relerror))
                        names_files_different.append(basename(in_file_1))

                        out_diffimages_filename = joinpathnames(args.tempdir, nameOutDiffImageFilesName %(i+1))
                        print("Output difference between images maps: \'%s\'..." %(basename(out_diffimages_filename)))

                        FileReader.writeImageArray(out_diffimages_filename, out_diffimages_array)

                        out_histo_filename = joinpathnames(args.tempdir, nameOutHistoFilesName%(i+1))
                        print("Compute and output the histograms of both images: \'%s\'..." %(basename(out_histo_filename)))

                        Histograms.plot_compare_histograms([in_image1_array, in_image2_array],
                                                           density_range=True,
                                                           isave_outfiles=True,
                                                           outfilename=out_histo_filename)
                        
        else:
            print("Images of different size: \'%s\' != \'%s\'" %(in_image1_array.shape, in_image2_array.shape))

            out_histo_filename = joinpathnames(args.tempdir, nameOutHistoFilesName%(i+1))
            print("Compute and output the histograms of both images: \'%s\'..." %(basename(out_histo_filename)))
            print("CHECK MANUALLY THE HISTOGRAMS WHETHER THE IMAGES ARE DIFFERENT")

            names_files_different.append(basename(in_file_1))

            Histograms.plot_compare_histograms([in_image1_array, in_image2_array],
                                               density_range=True,
                                               isave_outfiles=True,
                                               outfilename=out_histo_filename)
    #endfor

    if (len(names_files_different) == 0):
        print("\nGOOD: ALL IMAGE FILES ARE EQUAL...")
    else:
        print("\nWARNING: Found \'%s\' files that are different. Names of files: \'%s\'..." %(len(names_files_different),
                                                                                              names_files_different))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir1', type=str)
    parser.add_argument('inputdir2', type=str)
    parser.add_argument('--tempdir', type=str, default='.')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
