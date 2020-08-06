#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from common.constant import *
from common.workdirmanager import *
from dataloaders.imagefilereader import *
from collections import OrderedDict
import argparse



def main(args):

    workDirsManager     = GeneralDirManager(args.datadir)
    InputImagesPath     = workDirsManager.get_pathdir_exist(args.nameInputImagesRelPath)
    InputReferKeysPath  = workDirsManager.get_pathdir_exist(args.nameInputReferKeysRelPath)
    OutputRescaleFactorsFile = workDirsManager.get_pathfile_update(args.nameOutputRescaleFactorsFile)
    OutputOrigVoxelSizeFile  = workDirsManager.get_pathfile_update(args.nameOutputOrigVoxelSizeFile)

    listInputImagesFiles    = list_files_dir(InputImagesPath)
    listInputReferKeysFiles = list_files_dir(InputReferKeysPath)



    outdict_voxelSizes = OrderedDict()

    for i, in_image_file in enumerate(listInputImagesFiles):
        print("\nInput: \'%s\'..." %(basename(in_image_file)))

        in_voxel_size = DicomReader.get_image_voxelsize(in_image_file)
        print("Voxel Size: \'%s\'..." %(str(in_voxel_size)))

        in_referkey_file = listInputReferKeysFiles[i]
        outdict_voxelSizes[basename_file_noext(in_referkey_file)] = in_voxel_size
    #endfor


    # Save dictionary in file
    save_dictionary(OutputOrigVoxelSizeFile, outdict_voxelSizes)
    save_dictionary_csv(OutputOrigVoxelSizeFile.replace('.npy', '.csv'), outdict_voxelSizes)


    data = np.array(list(outdict_voxelSizes.values()))
    mean = np.mean(data, axis=0)
    print("\nMean value: \'%s\'..." %(mean))
    median = np.median(data, axis=0)
    print("Median value: \'%s\'..." %(median))


    if args.fixedRescaleRes:
        final_rescale_res = args.fixedRescaleRes
    else:
        #if not fixed scale specified, take median over dataset
        final_rescale_res = median
    print("Final rescaling resolution: \'%s\'..." %(str(final_rescale_res)))


    outdict_rescaleFactors = OrderedDict()

    for in_key_file, in_voxel_size in outdict_voxelSizes.items():
        print("\nInput Key file: \'%s\'..." % (in_key_file))

        rescale_factor = tuple(np.array(in_voxel_size) / np.array(final_rescale_res))
        print("Computed rescale factor: \'%s\'..." %(str(rescale_factor)))

        outdict_rescaleFactors[in_key_file] = rescale_factor
    #endfor


    # Save dictionary in file
    save_dictionary(OutputRescaleFactorsFile, outdict_rescaleFactors)
    save_dictionary_csv(OutputRescaleFactorsFile.replace('.npy', '.csv'), outdict_rescaleFactors)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--nameInputImagesRelPath', type=str, default=NAME_RAW_IMAGES_RELPATH)
    parser.add_argument('--nameInputReferKeysRelPath', type=str, default=NAME_REFERENCE_KEYS_RELPATH)
    parser.add_argument('--nameOutputOrigVoxelSizeFile', type=str, default='original_voxelSize.npy')
    parser.add_argument('--nameOutputRescaleFactorsFile', type=str, default=NAME_RESCALE_FACTOR_FILE)
    parser.add_argument('--fixedRescaleRes', type=str2tuple_float, default=FIXED_RESCALE_RESOL)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)