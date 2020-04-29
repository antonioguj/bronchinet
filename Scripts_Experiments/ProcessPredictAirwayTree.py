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
from OperationImages.OperationMasks import *
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameOutputBinaryMasksFiles = lambda in_name, thres: basename(in_name).replace('probmap','binmask_thres%s') %(str(thres).replace('.',''))
    # ---------- SETTINGS ----------


    workDirsManager       = WorkDirsManager(args.basedir)
    InputPosteriorsPath   = workDirsManager.getNameExistPath(args.nameInputPosteriorsRelPath)
    OutputBinaryMasksPath = workDirsManager.getNameNewPath  (args.nameOutputBinaryMasksRelPath)

    listInputPosteriorsFiles = findFilesDirAndCheck(InputPosteriorsPath)
    prefixPatternInputFiles  = getFilePrefixPattern(listInputPosteriorsFiles[0])

    if (args.attachCoarseAirwaysMask):
        InputCoarseAirwaysPath      = workDirsManager.getNameExistBaseDataPath(args.nameInputCoarseAirwaysRelPath)
        listInputCoarseAirwaysFiles = findFilesDirAndCheck(InputCoarseAirwaysPath)


    print("Compute \'%s\' Binary Masks from the Posteriors, using thresholding values: \'%s\'..." % (len(args.threshold_values),
                                                                                                     args.threshold_values))

    for i, in_posterior_file in enumerate(listInputPosteriorsFiles):
        print("\nInput: \'%s\'..." % (basename(in_posterior_file)))

        inout_posterior_array = FileReader.getImageArray(in_posterior_file)
        print("Original dims : \'%s\'..." % (str(inout_posterior_array.shape)))

        in_metadata_file = FileReader.getImageMetadataInfo(in_posterior_file)


        if (args.attachCoarseAirwaysMask):
            print("Attach Trachea and Main Bronchi mask to complete the computed Binary Masks...")
            in_coarseairways_file = findFileWithSamePrefixPattern(basename(in_posterior_file), listInputCoarseAirwaysFiles,
                                                                  prefix_pattern=prefixPatternInputFiles)
            print("Coarse Airways mask file: \'%s\'..." % (basename(in_coarseairways_file)))

            in_coarseairways_array = FileReader.getImageArray(in_coarseairways_file)


        for ithreshold in args.threshold_values:
            print("Compute Binary Masks thresholded to \'%s\'..." %(ithreshold))

            out_binarymask_array = ThresholdImages.compute(inout_posterior_array, ithreshold)

            if (args.attachCoarseAirwaysMask):
                out_binarymask_array = OperationBinaryMasks.merge_two_masks(out_binarymask_array, in_coarseairways_array) #isNot_intersect_masks=True)


            # Output predicted binary masks
            output_binarymask_file = joinpathnames(OutputBinaryMasksPath, nameOutputBinaryMasksFiles(in_posterior_file, ithreshold))
            print("Output: \'%s\', of dims \'%s\'..." % (basename(output_binarymask_file), str(out_binarymask_array.shape)))

            FileReader.writeImageArray(output_binarymask_file, out_binarymask_array, metadata=in_metadata_file)
        #endfor
    # endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('--nameInputPosteriorsRelPath', type=str, default=NAME_POSTERIORS_RELPATH)
    parser.add_argument('--nameInputCoarseAirwaysRelPath', type=str, default=NAME_RAWCOARSEAIRWAYS_RELPATH)
    parser.add_argument('--nameOutputBinaryMasksRelPath', type=str, default=NAME_PREDBINARYMASKS_RELPATH)
    parser.add_argument('--threshold_values', type=float, default=THRESHOLDPOST)
    parser.add_argument('--attachCoarseAirwaysMask', type=str2bool, default=ATTACHCOARSEAIRWAYSMASK)
    args = parser.parse_args()

    if type(args.threshold_values) in [int, float]:
        args.threshold_values = [args.threshold_values]

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)