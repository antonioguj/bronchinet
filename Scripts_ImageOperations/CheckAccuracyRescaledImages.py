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
from PlotsManager.Histograms import *
from collections import OrderedDict
import numpy as np
import argparse


LIST_CHECKS_COMPARE = ['volume_mask', 'histogram']
_max_relerror_volume_mask = 1.0e-02
_max_relerror_histogram   = 1.0e-03


def check_compare_equal_volume_mask(in_orig_array, in_resc_array, add_kwargs):
    print("Compare the volumes of original and rescaled masks...")

    orig_voxel_size = add_kwargs['orig_voxel_size']
    resc_voxel_size = add_kwargs['resc_voxel_size']

    num_voxels_orig_array = np.sum(in_orig_array)
    num_voxels_resc_array = np.sum(in_resc_array)

    volmask_orig_image = np.prod(orig_voxel_size) * num_voxels_orig_array
    volmask_resc_image = np.prod(resc_voxel_size) * num_voxels_resc_array

    rel_error_vols = abs(1.0 - volmask_resc_image/volmask_orig_image)
    if rel_error_vols <_max_relerror_volume_mask:
        print("Check GOOD: masks have similar volumes: \'%s\' == \'%s\'. Rel error: \'%s\'..."
              %(volmask_orig_image, volmask_resc_image, rel_error_vols))
        return True
    else:
        print("Check BAD: masks have different volumes: \'%s\' != \'%s\'. Rel error: \'%s\'..."
              %(volmask_orig_image, volmask_resc_image, rel_error_vols))
        return False


def check_compare_equal_histogram_bins(in_orig_array, in_resc_array, add_kwargs):
    print("Compare the histogram bins of original and rescaled images...")
    num_bins = 20

    orig_hist_data = Histograms.get_histogram_data(in_orig_array, num_bins=num_bins)
    resc_hist_data = Histograms.get_histogram_data(in_resc_array, num_bins=num_bins)

    orig_hist_data = orig_hist_data / float(in_orig_array.size)
    resc_hist_data = resc_hist_data / float(in_resc_array.size)

    diff_hists_data = abs(orig_hist_data - resc_hist_data)

    max_diff_hists_data = max(diff_hists_data)
    if max_diff_hists_data <_max_relerror_histogram:
        print("Check GOOD: images have similar histograms: \'max_diff_hists_data\' = \'%s\'" %(max_diff_hists_data))
        return True
    else:
        print("Check BAD: images have different histograms: \'max_diff_hists_data\' = \'%s\'" %(max_diff_hists_data))
        return False



def main(args):
    # ---------- SETTINGS ----------
    nameReferenceFilesRelPath  = 'RawImages/'
    nameOriginalVoxelSizesFile = 'RescalingData/original_vozelSize.npy'
    list_names_checks_compare  = args.type
    rescaled_voxel_size_same   = args.fixedRescaleRes
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(args.datadir)
    ReferenceFilesPath = workDirsManager.getNameExistPath(nameReferenceFilesRelPath)

    listInputOriginalFiles = findFilesDirAndCheck(args.inputoriginaldir)
    listInputRescaledFiles = findFilesDirAndCheck(args.inputrescaleddir)
    listReferenceFiles     = findFilesDirAndCheck(ReferenceFilesPath)

    filename_originalVoxelSizes = joinpathnames(args.datadir, nameOriginalVoxelSizesFile)
    dict_originalVoxelSizes     = readDictionary(filename_originalVoxelSizes)

    if (len(listInputOriginalFiles) != len(listInputRescaledFiles)):
       message = 'num files in dir 1 \'%s\', not equal to num files in dir 2 \'%i\'...' %(len(listInputOriginalFiles),
                                                                                          len(listInputRescaledFiles))
       CatchErrorException(message)


    dict_func_checks_compare = OrderedDict()

    for name_check_fun in list_names_checks_compare:
        if name_check_fun not in LIST_CHECKS_COMPARE:
            message = 'Check \'%s\' not yet implemented...' %(name_check_fun)
            CatchErrorException(message)
        else:
            if name_check_fun == 'volume_mask':
                new_func_checks_compare = check_compare_equal_volume_mask
            elif name_check_fun == 'histogram':
                new_func_checks_compare = check_compare_equal_histogram_bins
            else:
                new_func_checks_compare = None
            dict_func_checks_compare[name_check_fun] = new_func_checks_compare
    #endfor



    names_files_different = []

    for i, (in_orig_file, in_resc_file) in enumerate(zip(listInputOriginalFiles, listInputRescaledFiles)):
        print("\nInput: \'%s\'..." % (in_orig_file))
        print("And: \'%s\'..." % (in_resc_file))

        in_orig_image_array = FileReader.getImageArray(in_orig_file)
        in_resc_image_array = FileReader.getImageArray(in_resc_file)


        in_reference_file = listReferenceFiles[i]
        orig_voxel_size = dict_originalVoxelSizes[basenameNoextension(in_reference_file)]
        resc_voxel_size = rescaled_voxel_size_same
        dict_add_kwargs = {'orig_voxel_size': orig_voxel_size,
                           'resc_voxel_size': resc_voxel_size}


        all_checks_OK = True
        for func_name, func_checks_compare in dict_func_checks_compare.items():
            is_checkOK = func_checks_compare(in_orig_image_array, in_resc_image_array, dict_add_kwargs)
            if not is_checkOK:
                print("WRONG: Check \'%s\' is not correct..." %(func_name))
                names_files_different.append(in_orig_file)
            all_checks_OK *= is_checkOK
        #endfor

        if all_checks_OK:
            print("GOOD: All Checks are correct...")
    #endfor

    if (len(names_files_different) == 0):
        print("\nGOOD: ALL IMAGE FILES ARE EQUAL...")
    else:
        print("\nWARNING: Found \'%s\' files that are different. Names of files: \'%s\'..." %(len(names_files_different),
                                                                                              names_files_different))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('inputoriginaldir', type=str)
    parser.add_argument('inputrescaleddir', type=str)
    parser.add_argument('--type', nargs='+', default=['volume_mask'])
    parser.add_argument('--fixedRescaleRes', type=str2tuplefloat, default=FIXEDRESCALERES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)

