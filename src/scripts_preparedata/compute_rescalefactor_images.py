
from collections import OrderedDict
import numpy as np
import argparse

from common.constant import DATADIR, FIXED_RESCALE_RESOL, NAME_RAW_IMAGES_RELPATH, NAME_RESCALE_FACTORS_FILE, \
    NAME_REFERENCE_FILES_RELPATH
from common.functionutil import basename, basename_filenoext, list_files_dir, str2tuple_float_none, \
    save_dictionary, save_dictionary_csv
from common.workdirmanager import GeneralDirManager
from dataloaders.imagefilereader import DicomReader


def main(args):

    workdir_manager = GeneralDirManager(args.datadir)
    input_images_path = workdir_manager.get_pathdir_exist(args.name_input_images_relpath)
    input_reference_files_path = workdir_manager.get_pathdir_exist(args.name_input_reference_files_relpath)
    output_rescale_factors_file = workdir_manager.get_pathfile_update(args.name_output_rescale_factors_file)
    output_orig_voxelsize_file = workdir_manager.get_pathfile_update(args.name_output_original_voxelsize_file)
    list_input_images_files = list_files_dir(input_images_path)
    list_input_reference_files = list_files_dir(input_reference_files_path)

    outdict_voxel_sizes = OrderedDict()

    for i, in_image_file in enumerate(list_input_images_files):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))

        in_voxel_size = DicomReader.get_image_voxelsize(in_image_file)
        print("Voxel Size: \'%s\'..." % (str(in_voxel_size)))

        in_reference_file = list_input_reference_files[i]
        outdict_voxel_sizes[basename_filenoext(in_reference_file)] = in_voxel_size
    # endfor

    # Save computed original voxel sizes
    save_dictionary(output_orig_voxelsize_file, outdict_voxel_sizes)
    save_dictionary_csv(output_orig_voxelsize_file.replace('.npy', '.csv'), outdict_voxel_sizes)

    data_voxel_sizes = np.array(list(outdict_voxel_sizes.values()))
    mean_voxel_size = np.mean(data_voxel_sizes, axis=0)
    print("\nMean value: \'%s\'..." % (mean_voxel_size))
    median_voxel_size = np.median(data_voxel_sizes, axis=0)
    print("Median value: \'%s\'..." % (median_voxel_size))

    # *****************************************************

    if args.fixed_rescale_resol:
        final_rescale_res = args.fixed_rescale_resol
    else:
        # if not fixed scale specified, take median over dataset
        final_rescale_res = median_voxel_size
    print("Final rescaling resolution: \'%s\'..." % (str(final_rescale_res)))

    outdict_rescale_factors = OrderedDict()

    for in_key_file, in_voxel_size in outdict_voxel_sizes.items():
        print("\nInput Key file: \'%s\'..." % (in_key_file))

        rescale_factor = tuple(np.array(in_voxel_size) / np.array(final_rescale_res))
        print("Computed rescale factor: \'%s\'..." % (str(rescale_factor)))

        outdict_rescale_factors[in_key_file] = rescale_factor
    # endfor

    # Save computed rescale factors
    save_dictionary(output_rescale_factors_file, outdict_rescale_factors)
    save_dictionary_csv(output_rescale_factors_file.replace('.npy', '.csv'), outdict_rescale_factors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--fixed_rescale_resol', type=str2tuple_float_none, default=FIXED_RESCALE_RESOL)
    parser.add_argument('--name_input_images_relpath', type=str, default=NAME_RAW_IMAGES_RELPATH)
    parser.add_argument('--name_output_rescale_factors_file', type=str, default=NAME_RESCALE_FACTORS_FILE)
    parser.add_argument('--name_output_original_voxelsize_file', type=str, default='original_voxelSize.npy')
    parser.add_argument('--name_input_reference_files_relpath', type=str, default=NAME_REFERENCE_FILES_RELPATH)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" % (key, value))
    main(args)
