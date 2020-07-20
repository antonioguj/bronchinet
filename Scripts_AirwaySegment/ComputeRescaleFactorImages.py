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
import argparse



def main(args):

    workDirsManager     = WorkDirsManager(args.datadir)
    InputImagesPath     = workDirsManager.getNameExistPath(args.nameInputImagesRelPath)
    InputReferKeysPath  = workDirsManager.getNameExistPath(args.nameInputReferKeysRelPath)
    OutputRescaleFactorsFile = workDirsManager.getNameNewUpdateFile(args.nameOutputRescaleFactorsFile)
    OutputOrigVoxelSizeFile  = workDirsManager.getNameNewUpdateFile(args.nameOutputOrigVoxelSizeFile)

    listInputImagesFiles    = findFilesDirAndCheck(InputImagesPath)
    listInputReferKeysFiles = findFilesDirAndCheck(InputReferKeysPath)



    outdict_voxelSizes = OrderedDict()

    for i, in_image_file in enumerate(listInputImagesFiles):
        print("\nInput: \'%s\'..." %(basename(in_image_file)))

        in_voxel_size = DICOMreader.get_image_voxel_size(in_image_file)
        print("Voxel Size: \'%s\'..." %(str(in_voxel_size)))

        in_referkey_file = listInputReferKeysFiles[i]
        outdict_voxelSizes[basenameNoextension(in_referkey_file)] = in_voxel_size
    #endfor


    # Save dictionary in file
    saveDictionary(OutputOrigVoxelSizeFile, outdict_voxelSizes)
    saveDictionary_csv(OutputOrigVoxelSizeFile.replace('.npy','.csv'), outdict_voxelSizes)


    data = np.array(outdict_voxelSizes.values())
    mean = np.mean(data, axis=0)
    print("\nMean value: \'%0.6f\'..." %(mean))
    median = np.median(data, axis=0)
    print("Median value: \'%0.6f\'..." %(median))


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
    saveDictionary(OutputRescaleFactorsFile, outdict_rescaleFactors)
    saveDictionary_csv(OutputRescaleFactorsFile.replace('.npy','.csv'), outdict_rescaleFactors)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--nameInputImagesRelPath', type=str, default=NAME_RAWIMAGES_RELPATH)
    parser.add_argument('--nameInputReferKeysRelPath', type=str, default=NAME_REFERKEYS_RELPATH)
    parser.add_argument('--nameOutputOrigVoxelSizeFile', type=str, default='original_voxelSize.npy')
    parser.add_argument('--nameOutputRescaleFactorsFile', type=str, default=NAME_RESCALEFACTOR_FILE)
    parser.add_argument('--fixedRescaleRes', type=str2tuplefloatOrNone, default=FIXEDRESCALERES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)