#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from DataLoaders.FileReaders import *
from PlotsManager.Histograms import *
import argparse



def main(args):
    # ---------- SETTINGS ----------
    InputPath1       = args.inputdir1
    InputPath2       = args.inputdir2
    namesInputFiles1 = '*.nii.gz'
    namesInputFiles2 = '*.nii.gz'
    max_rel_error    = 1.0e-06
    nameOutFilesName = 'out_histo_image%s.png'
    # ---------- SETTINGS ----------


    listInputFiles1 = findFilesDirAndCheck(InputPath1, namesInputFiles1)
    listInputFiles2 = findFilesDirAndCheck(InputPath2, namesInputFiles2)

    if (len(listInputFiles1) != len(listInputFiles2)):
        message = 'num files in dir 1 \'%s\', not equal to num files in dir 2 \'%i\'...' %(len(listInputFiles1), len(listInputFiles2))
        CatchErrorException(message)

    if not isExistdir(args.tempdir):
        makedir(args.tempdir)



    names_files_different = []
    names_files_perhapsdiff_showhistogram = []

    for i, (in_file_1, in_file_2) in enumerate(zip(listInputFiles1, listInputFiles2)):
        print("\nInput 1: \'%s\'..." % (in_file_1))
        print("Input 2: \'%s\'..." % (in_file_2))

        in_img1_array = FileReader.getImageArray(in_file_1)
        in_img2_array = FileReader.getImageArray(in_file_2)

        #print("Values in input array 1: %s..." % (np.unique(in_img1_array)))
        #print("Values in input array 2: %s..." % (np.unique(in_img2_array)))


        if (in_img1_array.shape == in_img2_array.shape):
            print("Images of same size: \'%s\'..." %(str(in_img1_array.shape)))

            res_voxels_equal = np.array_equal(in_img1_array, in_img2_array)

            if res_voxels_equal:
                print("GOOD: Images are equal voxelwise...")
            else:
                message = "Images 1 and 2 are not equal voxelwise. Analyse global magns..."
                CatchWarningException(message)

                abs_error_array = abs(in_img1_array - in_img2_array)

                num_voxels_diff = np.count_nonzero(abs_error_array)
                num_voxels_1_nonzero = np.count_nonzero(in_img1_array)
                rel_num_voxels_diff = (num_voxels_diff / float(num_voxels_1_nonzero)) * 100
                print("Num voxels different \'%s\' out of total \'%s\'. Rel percentage \'%s\'..." %(num_voxels_diff,
                                                                                                    abs_error_array.size,
                                                                                                    rel_num_voxels_diff))
                mean_abs_error = abs(np.mean(abs_error_array))
                mean_in_img1 = abs(np.mean(in_img1_array))
                mean_rel_error = (mean_abs_error / float(mean_in_img1)) * 100
                print("Absolute / relative error between images: \'%s\' / \'%s\'..." %(mean_abs_error, mean_rel_error))

                if mean_rel_error < max_rel_error:
                    print("GOOD. Error lower than tolerance \'%s\'. Images can be considered equal..." %(max_rel_error))
                else:
                    names_files_different.append(basename(in_file_1))
                    message = "Error larger than tolerance \'%s\'. Images are different..." %(max_rel_error)
                    CatchWarningException(message)


                print("Compute the histograms of both images...")
                hist_outfilename = joinpathnames(args.tempdir, nameOutFilesName % (i+1))

                Histograms.plot_compare_histograms([in_img1_array, in_img2_array],
                                                   isave_outfiles=True,
                                                   outfilename=hist_outfilename,
                                                   show_percen_yaxis=True)

                names_files_perhapsdiff_showhistogram.append(basename(in_file_1))
                message = "Do not know whether images are different. Check saved histogram in: \'%s\'..." %(hist_outfilename)
                CatchWarningException(message)

        else:
            print("Images of different size: \'%s\' != \'%s\'" %(in_img1_array.shape, in_img2_array.shape))
            print("Compute the histograms of both images...")

            hist_outfilename = joinpathnames(args.tempdir, nameOutFilesName%(i+1))

            Histograms.plot_compare_histograms([in_img1_array, in_img2_array],
                                               isave_outfiles= True,
                                               outfilename= hist_outfilename,
                                               show_percen_yaxis= True)

            names_files_perhapsdiff_showhistogram.append(basename(in_file_1))
            message = "Do not know whether images are different. Check saved histogram in: \'%s\'..." % (hist_outfilename)
            CatchWarningException(message)
    #endfor


    if (len(names_files_different) == 0):
        print("\nGOOD: ALL IMAGE FILES ARE EQUAL...")
    else:
        print("\nWARNING: Found \'%s\' files that are different. Names of files: \'%s\'..." %(len(names_files_different),
                                                                                              names_files_different))
    if (len(names_files_perhapsdiff_showhistogram) != 0):
        print("\nFound \'%s\' files that perhaps are different. Names of files: \'%s\'..." %(len(names_files_perhapsdiff_showhistogram),
                                                                                             names_files_perhapsdiff_showhistogram))
        print("Check histograms stored in: \'%s\'..." % (args.tempdir))



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
