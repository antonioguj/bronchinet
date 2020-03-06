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



    out_dictVoxelSizes = OrderedDict()

    for i, in_image_file in enumerate(listInputImagesFiles):
        print("\nInput: \'%s\'..." %(basename(in_image_file)))

        in_voxel_size = DICOMreader.getImageVoxelSize(in_image_file)
        print("Voxel Size: \'%s\'..." %(str(in_voxel_size)))

        in_referkey_file = listInputReferKeysFiles[i]
        out_dictVoxelSizes[basenameNoextension(in_referkey_file)] = in_voxel_size
    #endfor


    # Save dictionary in file
    saveDictionary(OutputOrigVoxelSizeFile, out_dictVoxelSizes)
    saveDictionary_csv(OutputOrigVoxelSizeFile.replace('.npy','.csv'), out_dictVoxelSizes)


    data = np.array(out_dictVoxelSizes.values())
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


    out_dictRescaleFactors = OrderedDict()

    for key, value in out_dictVoxelSizes.iteritems():
        print("\nKey: \'%s\'..." %(key))

        rescale_factor = tuple(np.array(value) / np.array(final_rescale_res))
        print("Computed rescale factor: \'%s\'..." %(str(rescale_factor)))

        out_dictRescaleFactors[key] = rescale_factor
    #endfor


    # Save dictionary in file
    saveDictionary(OutputRescaleFactorsFile, out_dictRescaleFactors)
    saveDictionary_csv(OutputRescaleFactorsFile.replace('.npy','.csv'), out_dictRescaleFactors)



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
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)