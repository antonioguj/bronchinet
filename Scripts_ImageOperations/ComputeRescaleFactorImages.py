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
    # ---------- SETTINGS ----------
    nameInputImagesRelPath     = 'RawImages/'
    nameOutputDataRelPath      = 'RescalingData/'
    nameOrigVoxelSize_FileNpy  = 'original_vozelSize.npy'
    nameOrigVoxelSize_FileCsv  = 'original_vozelSize.csv'
    nameRescaleFactors_FileNpy = 'rescaleFactors_images.npy'
    nameRescaleFactors_FileCsv = 'rescaleFactors_images.csv'
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(args.datadir)
    InputImagesPath = workDirsManager.getNameExistPath(nameInputImagesRelPath)
    OutputDataPath  = workDirsManager.getNameNewPath  (nameOutputDataRelPath)

    listInputImageFiles = findFilesDirAndCheck(InputImagesPath)


    dict_voxelSizes = OrderedDict()

    for i, in_image_file in enumerate(listInputImageFiles):
        print("\nInput: \'%s\'..." %(basename(in_image_file)))

        in_voxel_size = DICOMreader.getVoxelSize(in_image_file)
        print("Voxel Size: \'%s\'..." %(str(in_voxel_size)))

        dict_voxelSizes[filenamenoextension(in_image_file)] = in_voxel_size
    #endfor


    # Save dictionary in file
    nameoutfile = joinpathnames(OutputDataPath, nameOrigVoxelSize_FileNpy)
    saveDictionary(nameoutfile, dict_voxelSizes)
    nameoutfile = joinpathnames(OutputDataPath, nameOrigVoxelSize_FileCsv)
    saveDictionary_csv(nameoutfile, dict_voxelSizes)


    data = np.array(dict_voxelSizes.values())
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


    dict_rescaleFactors = OrderedDict()

    for key, value in dict_voxelSizes.iteritems():
        print("\nKey: \'%s\'..." %(key))

        rescale_factor = tuple(np.array(value) / np.array(final_rescale_res))
        print("Computed rescale factor: \'%s\'..." %(str(rescale_factor)))

        dict_rescaleFactors[key] = rescale_factor
    #endfor


    # Save dictionary in file
    nameoutfile = joinpathnames(OutputDataPath, nameRescaleFactors_FileNpy)
    saveDictionary(nameoutfile, dict_rescaleFactors)
    nameoutfile = joinpathnames(OutputDataPath, nameRescaleFactors_FileCsv)
    saveDictionary_csv(nameoutfile, dict_rescaleFactors)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=DATADIR)
    parser.add_argument('--fixedRescaleRes', type=str2tuplefloat, default=FIXEDRESCALERES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)