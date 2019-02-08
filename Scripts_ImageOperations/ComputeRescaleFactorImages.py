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
    nameInputRelPath = 'RawImages'

    nameInputFiles = '*.dcm'

    nameOrigVoxelSize_FileNpy = 'original_vozelSize.npy'
    nameOrigVoxelSize_FileCsv = 'original_vozelSize.csv'
    nameRescaleFactors_FileNpy = 'rescaleFactors_images_0.6x0.6x0.6.npy'
    nameRescaleFactors_FileCsv = 'rescaleFactors_images_0.6x0.6x0.6.csv'
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(BASEDIR)
    BaseDataPath = workDirsManager.getNameBaseDataPath()
    InputPath = workDirsManager.getNameExistPath(BaseDataPath, nameInputRelPath)

    listInputFiles = findFilesDirAndCheck(InputPath, nameInputFiles)


    dict_voxelSizes = OrderedDict()

    for i, in_file in enumerate(listInputFiles):
        print("\nInput: \'%s\'..." %(basename(in_file)))

        voxel_size = DICOMreader.getVoxelSize(in_file)
        print("Voxel Size: \'%s\'..." %(str(voxel_size)))

        dict_voxelSizes[filenamenoextension(in_file)] = voxel_size
    #endfor


    # Save dictionary in file
    nameoutfile = joinpathnames(BaseDataPath, nameOrigVoxelSize_FileNpy)
    saveDictionary(nameoutfile, dict_voxelSizes)
    nameoutfile = joinpathnames(BaseDataPath, nameOrigVoxelSize_FileCsv)
    saveDictionary_csv(nameoutfile, dict_voxelSizes)


    data = np.array(dict_voxelSizes.values())
    mean = np.mean(data, axis=0)
    print("Mean value: \'%s\'..." %(mean))
    median = np.median(data, axis=0)
    print("Median value: \'%s\'..." %(median))


    if args.fixedRescaleRes:
        final_rescale_res = args.fixedRescaleRes
    else:
        #if not fixed scale specified, take median over dataset
        final_rescale_res = median
    print("Final aimed resolution: \'%s\'..." %(str(final_rescale_res)))

    dict_rescaleFactors = OrderedDict()

    for key, value in dict_voxelSizes.iteritems():
        print("\nKey: \'%s\'..." %(key))

        rescale_factor = tuple(np.array(value) / np.array(final_rescale_res))
        print("Computed rescale factor: \'%s\'..." %(str(rescale_factor)))

        dict_rescaleFactors[key] = rescale_factor
    #endfor


    # Save dictionary in file
    nameoutfile = joinpathnames(BaseDataPath, nameRescaleFactors_FileNpy)
    saveDictionary(nameoutfile, dict_rescaleFactors)
    nameoutfile = joinpathnames(BaseDataPath, nameRescaleFactors_FileCsv)
    saveDictionary_csv(nameoutfile, dict_rescaleFactors)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--fixedRescaleRes', type=str2tuplefloat, default=FIXEDRESCALERES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)