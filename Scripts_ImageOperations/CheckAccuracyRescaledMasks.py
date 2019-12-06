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
from collections import OrderedDict
import numpy as np
import argparse


LIST_CHECKS_ACCURACY = ['mask_volume']
_max_relerror = 1.0e-06


def check_equal_mask_volume(in_orig_array, in_resc_array, **kwargs):
    orig_voxel_size = kwargs['orig_voxel_size']
    resc_voxel_size = kwargs['resc_voxel_size']

    num_voxels_orig_array = np.sum(in_orig_array)
    num_voxels_resc_array = np.sum(in_resc_array)

    volmask_orig_image = np.prod(orig_voxel_size) * num_voxels_orig_array
    volmask_resc_image = np.prod(resc_voxel_size) * num_voxels_resc_array

    rel_error_vols = abs(1.0 - volmask_resc_image/volmask_resc_image)
    if rel_error_vols <_max_relerror:
        print("Original and Rescaled masks have similar volumes: \'%s\' == \'%s\'..." %(volmask_orig_image, volmask_resc_image))
        return True
    else:
        print("Original and Rescaled masks have different volumes: \'%s\' != \'%s\'..." %(volmask_orig_image, volmask_resc_image))
        return False



def main(args):
    # ---------- SETTINGS ----------
    nameReferenceFilesRelPath  = 'RawImages/'
    nameOriginalVoxelSizesFile = 'RescalingData/original_vozelSize.npy'
    rescaled_voxel_size_same = (0.6, 0.55078125, 0.55078125)
    # ---------- SETTINGS ----------


    ReferenceFilesPath = workDirsManager.getNameExistBaseDataPath(nameReferenceFilesRelPath)

    listInputOriginalFiles = findFilesDirAndCheck(args.inputoriginaldir)
    listInputRescaledFiles = findFilesDirAndCheck(args.inputrescaleddir)
    listReferenceFiles     = findFilesDirAndCheck(ReferenceFilesPath)

    filename_originalVoxelSizes = joinpathnames(args.datadir, nameOriginalVoxelSizesFile)
    dict_originalVoxelSizes     = readDictionary(filename_originalVoxelSizes)

    if (len(listInputOriginalFiles) != len(listInputRescaledFiles)):
       message = 'num files in dir 1 \'%s\', not equal to num files in dir 2 \'%i\'...' %(len(listInputOriginalFiles),
                                                                                          len(listInputRescaledFiles))
       CatchErrorException(message)


    dict_func_check_accuracy = OrderedDict()

    for check_fun_name in args.listChecksAccuracy:
        if check_fun_name not in LIST_CHECKS_ACCURACY:
            message = 'Check \'%s\' not yet implemented...' %(check_fun_name)
            CatchErrorException(message)
        else:
            if check_fun_name == 'mask_volume':
                dict_func_check_accuracy[check_fun_name] = check_equal_mask_volume
    #endfor



    names_files_different = []

    for i, (in_orig_file, in_resc_file) in enumerate(zip(listInputOriginalFiles, listInputRescaledFiles)):
        print("\nInput: \'%s\'..." % (in_orig_file))
        print("And: \'%s\'..." % (in_resc_file))

        in_orig_image_array = FileReader.getImageArray(in_orig_file)
        in_resc_image_array = FileReader.getImageArray(in_resc_file)


        in_reference_file = listReferenceFiles[i]
        orig_voxel_size = dict_originalVoxelSizes[filenamenoextension(in_reference_file)]
        resc_voxel_size = rescaled_voxel_size_same
        dist_add_kwargs = {'orig_voxel_size': orig_voxel_size,
                           'resc_voxel_size': resc_voxel_size}


        for key, func_check_accuracy in dictionary.items():
            is_checkOK = func_check_accuracy(in_orig_image_array, in_resc_image_array, dist_add_kwargs)
            if not is_checkOK:
                print("Check \'%s\' is correct..." %(is_checkOK))
                names_files_different.append(in_orig_file)
                break
        #endfor

        print("All Checks are correct...")
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
    parser.add_argument('--listChecksAccuracy', type=list, nargs='+', default=['mask_volume'])
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)

