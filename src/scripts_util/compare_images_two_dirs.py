
from collections import OrderedDict
import numpy as np
import argparse

from common.functionutil import makedir, join_path_names, is_exist_dir, list_files_dir, basename
from common.exceptionmanager import catch_error_exception
from dataloaders.imagefilereader import ImageFileReader
from imageoperators.imageoperator import MorphoOpenMask
# from imageoperators.maskoperator import MaskOperator
from plotting.histogram import Histogram


LIST_CHECKS_FURTHER_COMPARE = ['volume_mask', 'histogram']
_max_relerror = 1.0e-06
_max_relerror_volume_mask = 1.0e-02
_max_relerror_histogram = 1.0e-03


def check_compare_equal_volume_mask(in_image_1: np.ndarray, in_image_2: np.ndarray, **kwargs):
    print("Compare the volumes of original and rescaled masks...")
    img1_voxel_size = kwargs['img1_voxel_size']
    img2_voxel_size = kwargs['img2_voxel_size']

    num_voxels_image1 = np.sum(in_image_1)
    num_voxels_image2 = np.sum(in_image_2)

    volmask_image1 = np.prod(img1_voxel_size) * num_voxels_image1
    volmask_image2 = np.prod(img2_voxel_size) * num_voxels_image2

    rel_error_vols = abs(1.0 - volmask_image2 / volmask_image1)
    if rel_error_vols < _max_relerror_volume_mask:
        print("GOOD: masks have similar volumes: \'%s\' == \'%s\'. Rel error: \'%s\'..."
              % (volmask_image1, volmask_image2, rel_error_vols))
        return True
    else:
        print("BAD: masks have different volumes: \'%s\' != \'%s\'. Rel error: \'%s\'..."
              % (volmask_image1, volmask_image2, rel_error_vols))
        return False


def check_compare_equal_histogram_bins(in_image_1: np.ndarray, in_image_2: np.ndarray, **kwargs):
    print("Compare the histogram bins of original and rescaled images...")
    num_bins = 20

    img1_hist_data = Histogram.get_histogram_data(in_image_1, num_bins=num_bins)
    img2_hist_data = Histogram.get_histogram_data(in_image_2, num_bins=num_bins)

    img1_hist_data = img1_hist_data / float(in_image_1.size)
    img2_hist_data = img2_hist_data / float(in_image_2.size)

    diff_hists_data = abs(img1_hist_data - img2_hist_data)

    max_diff_hists_data = max(diff_hists_data)
    if max_diff_hists_data < _max_relerror_histogram:
        print("Check GOOD: images have similar histograms: \'max_diff_hists_data\' = \'%s\'" % (max_diff_hists_data))
        return True
    else:
        print("Check BAD: images have different histograms: \'max_diff_hists_data\' = \'%s\'" % (max_diff_hists_data))
        return False


def main(args):

    # SETTINGS
    name_output_diff_image_files_name = 'out_absdiffimgs_image%s.nii.gz'
    name_output_histo_files_name = 'out_histogram_image%s.png'
    # --------

    list_input_files_1 = list_files_dir(args.inputdir_1)
    list_input_files_2 = list_files_dir(args.inputdir_2)

    if len(list_input_files_1) != len(list_input_files_2):
        message = 'num files in dir 1 \'%s\', not equal to num files in dir 2 \'%i\'...' \
                  % (len(list_input_files_1), len(list_input_files_2))
        catch_error_exception(message)

    if not is_exist_dir(args.tempdir):
        makedir(args.tempdir)

    dict_func_checks_further_compare = OrderedDict()
    if args.type_checks_further:
        for name_check_fun in args.type_checks_further:
            if name_check_fun not in LIST_CHECKS_FURTHER_COMPARE:
                message = 'Check \'%s\' not yet implemented...' % (name_check_fun)
                catch_error_exception(message)
            else:
                if name_check_fun == 'volume_mask':
                    new_func_checks_compare = check_compare_equal_volume_mask
                elif name_check_fun == 'histogram':
                    new_func_checks_compare = check_compare_equal_histogram_bins
                else:
                    new_func_checks_compare = None
                dict_func_checks_further_compare[name_check_fun] = new_func_checks_compare
        # endfor

    # *****************************************************

    names_files_different = []

    for i, (in_file_1, in_file_2) in enumerate(zip(list_input_files_1, list_input_files_2)):
        print("\nInput: \'%s\'..." % (in_file_1))
        print("And: \'%s\'..." % (in_file_2))

        in_image_1 = ImageFileReader.get_image(in_file_1)
        in_image_2 = ImageFileReader.get_image(in_file_2)

        # in_image_1 = MaskOperator.binarise(in_image_1)
        # in_image_2 = MaskOperator.binarise(in_image_2)

        if in_image_1.shape == in_image_2.shape:
            is_images_equal_voxelwise = np.array_equal(in_image_1, in_image_2)

            if is_images_equal_voxelwise:
                print("GOOD: Images are equal voxelwise...")
            else:
                print("WARNING: Images 1 and 2 are not equal...")
                print("Do morphological opening of difference between images to remove strayed voxels due to noise...")

                out_diffimages = abs(in_image_1 - in_image_2)

                out_diffimages = MorphoOpenMask.compute(out_diffimages)

                num_voxels_diffimages = np.count_nonzero(out_diffimages)

                if num_voxels_diffimages == 0:
                    print("GOOD: After a morphological opening, Images are equal...")
                else:
                    print("WARNING: After a morphological opening, Images 1 and 2 are still not equal...")
                    print("Analyse global error magnitudes of the difference between images...")

                    num_voxels_nonzero_image1 = np.count_nonzero(in_image_1)
                    relerror_voxels_diffimages = (num_voxels_diffimages / float(num_voxels_nonzero_image1))

                    print("Num voxels different between images \'%s\' out of total non-zero voxels in image 1 \'%s\'. "
                          "Relative error \'%s\'..."
                          % (num_voxels_diffimages, num_voxels_nonzero_image1, relerror_voxels_diffimages))

                    absmean_intens_diffimages = abs(np.mean(out_diffimages))
                    absmean_intens_image1 = abs(np.mean(in_image_1))
                    relerror_intens_diffimages = (absmean_intens_diffimages / absmean_intens_image1)
                    print("Mean value of Intensity of difference between images \'%s\'. Relative error \'%s\'..."
                          % (absmean_intens_diffimages, relerror_intens_diffimages))

                    if relerror_intens_diffimages < _max_relerror:
                        print("GOOD. Relative error \'%s\' lower than tolerance \'%s\'. Images are considered equal..."
                              % (relerror_intens_diffimages, _max_relerror))
                    else:
                        print("WARNING. Relative error \'%s\' larger than tolerance \'%s\'. Images are different..."
                              % (relerror_intens_diffimages, _max_relerror))
                        names_files_different.append(basename(in_file_1))

                        out_diffimages_filename = join_path_names(args.tempdir,
                                                                  name_output_diff_image_files_name % (i + 1))
                        print("Output difference between images maps: \'%s\'..." % (basename(out_diffimages_filename)))

                        out_histo_filename = join_path_names(args.tempdir,
                                                             name_output_histo_files_name % (i + 1))
                        print("Output the histograms of both images: \'%s\'..." % (basename(out_histo_filename)))

                        Histogram.plot_compare_histograms([in_image_1, in_image_2],
                                                          density_range=True,
                                                          is_save_outfiles=True,
                                                          outfilename=out_histo_filename)
        else:
            print("Images of different size: \'%s\' != \'%s\'" % (in_image_1.shape, in_image_2.shape))

            out_histo_filename = join_path_names(args.tempdir, name_output_histo_files_name % (i + 1))
            print("Output the histograms of both images: \'%s\'..." % (basename(out_histo_filename)))
            print("CHECK MANUALLY THE HISTOGRAMS WHETHER THE IMAGES ARE DIFFERENT")

            names_files_different.append(basename(in_file_1))

            Histogram.plot_compare_histograms([in_image_1, in_image_2],
                                              num_bins=100,
                                              density_range=True,
                                              is_save_outfiles=True,
                                              outfilename=out_histo_filename)

            if len(dict_func_checks_further_compare) > 0:
                print("Do further checks to compare images of different size...")
                img1_voxel_size = ImageFileReader.get_image_voxelsize(in_file_1)
                img2_voxel_size = ImageFileReader.get_image_voxelsize(in_file_2)

                for func_name, func_checks_compare in dict_func_checks_further_compare.items():
                    is_check_ok = func_checks_compare(in_image_1, in_image_2,
                                                      img1_voxel_size=img1_voxel_size,
                                                      img2_voxel_size=img2_voxel_size)
                    if is_check_ok:
                        print("GOOD: Check \'%s\' is correct..." % (func_name))
                    else:
                        print("WRONG: Check \'%s\' is not correct..." % (func_name))
                # endfor
    # endfor

    if len(names_files_different) == 0:
        print("\nGOOD: ALL IMAGE FILES ARE EQUAL...")
    else:
        print("\nWARNING: Found \'%s\' files that are different. Names of files: \'%s\'..."
              % (len(names_files_different), names_files_different))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputdir_1', type=str)
    parser.add_argument('inputdir_2', type=str)
    parser.add_argument('--tempdir', type=str, default='temp_compare/')
    parser.add_argument('--type_checks_further', nargs='+', default=['volume_mask'])
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))

    main(args)
