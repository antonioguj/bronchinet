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
from OperationImages.OperationMasks import MorphoOpenMasks
from PlotsManager.Histograms import *
from collections import OrderedDict
import argparse

_max_relerror = 1.0e-06



LIST_CHECKS_FURTHER_COMPARE = ['volume_mask', 'histogram']
_max_relerror_volume_mask = 1.0e-02
_max_relerror_histogram   = 1.0e-03


# ------------------------------------------------
def check_compare_equal_volume_mask(in_image1_array, in_image2_array, **kwargs):
    print("Compare the volumes of original and rescaled masks...")
    img1_voxel_size = kwargs['img1_voxel_size']
    img2_voxel_size = kwargs['img2_voxel_size']

    num_voxels_image1_array = np.sum(in_image1_array)
    num_voxels_image2_array = np.sum(in_image2_array)

    volmask_image1 = np.prod(img1_voxel_size) * num_voxels_image1_array
    volmask_image2 = np.prod(img2_voxel_size) * num_voxels_image2_array

    rel_error_vols = abs(1.0 - volmask_image2/volmask_image1)
    if rel_error_vols <_max_relerror_volume_mask:
        print("GOOD: masks have similar volumes: \'%s\' == \'%s\'. Rel error: \'%s\'..." %(volmask_image1, volmask_image2, rel_error_vols))
        return True
    else:
        print("BAD: masks have different volumes: \'%s\' != \'%s\'. Rel error: \'%s\'..."%(volmask_image1, volmask_image2, rel_error_vols))
        return False


# ------------------------------------------------
def check_compare_equal_histogram_bins(in_image1_array, in_image2_array, **kwargs):
    print("Compare the histogram bins of original and rescaled images...")
    num_bins = 20

    img1_hist_data = Histograms.get_histogram_data(in_image1_array, num_bins=num_bins)
    img2_hist_data = Histograms.get_histogram_data(in_image2_array, num_bins=num_bins)

    img1_hist_data = img1_hist_data / float(in_image1_array.size)
    img2_hist_data = img2_hist_data / float(in_image2_array.size)

    diff_hists_data = abs(img1_hist_data - img2_hist_data)

    max_diff_hists_data = max(diff_hists_data)
    if max_diff_hists_data <_max_relerror_histogram:
        print("Check GOOD: images have similar histograms: \'max_diff_hists_data\' = \'%s\'" %(max_diff_hists_data))
        return True
    else:
        print("Check BAD: images have different histograms: \'max_diff_hists_data\' = \'%s\'" %(max_diff_hists_data))
        return False




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


    if args.type_checks_further:
        dict_func_checks_further_compare = OrderedDict()

        for name_check_fun in args.type_checks_further:
            if name_check_fun not in LIST_CHECKS_FURTHER_COMPARE:
                message = 'Check \'%s\' not yet implemented...' %(name_check_fun)
                CatchErrorException(message)
            else:
                if name_check_fun == 'volume_mask':
                    new_func_checks_compare = check_compare_equal_volume_mask
                elif name_check_fun == 'histogram':
                    new_func_checks_compare = check_compare_equal_histogram_bins
                else:
                    new_func_checks_compare = None
                dict_func_checks_further_compare[name_check_fun] = new_func_checks_compare
        #endfor



    names_files_different = []

    for i, (in_file_1, in_file_2) in enumerate(zip(listInputFiles1, listInputFiles2)):
        print("\nInput: \'%s\'..." % (in_file_1))
        print("And: \'%s\'..." % (in_file_2))

        in_image1_array = FileReader.getImageArray(in_file_1)
        in_image2_array = FileReader.getImageArray(in_file_2)

        # # Compare files metadata (header info)
        # if (filenameextension(in_file_1) == filenameextension(in_file_2)):
        #     in_img1_metadata = FileReader.getImageMetadataInfo(in_file_1)
        #     in_img2_metadata = FileReader.getImageMetadataInfo(in_file_2)
        #
        #     if (in_img1_metadata == in_img2_metadata):
        #         print("GOOD: Images have equal metadata (header info)...")
        #     else:
        #         print("WARNING: Images have different metadata (header info)...")


        if (in_image1_array.shape == in_image2_array.shape):
            is_arrays_equal_voxelwise = np.array_equal(in_image1_array, in_image2_array)

            if is_arrays_equal_voxelwise:
                print("GOOD: Images are equal voxelwise...")
            else:
                print("WARNING: Images 1 and 2 are not equal...")
                print("Do morphological opening of difference between images to remove strayed voxels due to noise...")

                out_diffimages_array = abs(in_image1_array - in_image2_array)

                out_diffimages_array = MorphoOpenMasks.compute(out_diffimages_array)

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

                        #FileReader.writeImageArray(out_diffimages_filename, out_diffimages_array)

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

            if len(dict_func_checks_further_compare) > 0:
                print('Do further checks to compare images of different size...')
                img1_voxel_size = FileReader.getImageVoxelSize(in_file_1)
                img2_voxel_size = FileReader.getImageVoxelSize(in_file_2)

                for func_name, func_checks_compare in dict_func_checks_further_compare.items():
                    is_check_OK = func_checks_compare(in_image1_array, in_image2_array,
                                                      img1_voxel_size=img1_voxel_size,
                                                      img2_voxel_size=img2_voxel_size)
                    if is_check_OK:
                        print("GOOD: Check \'%s\' is correct..." % (func_name))
                    else:
                        print("WRONG: Check \'%s\' is not correct..." %(func_name))
                #endfor
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
    parser.add_argument('--tempdir', type=str, default='temp_compare/')
    parser.add_argument('--type_checks_further', nargs='+', default=['volume_mask'])
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
