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
from Preprocessing.OperationImages import *
from Preprocessing.OperationMasks import *
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameInputPredictionsRelPath = args.inputpredictiondir
    nameInputReferMasksRelPath  = 'Airways_Proc/'
    nameInputRoiMasksRelPath    = 'Lungs_Proc/'
    nameInputPredictionsFiles   = '*_probmap.nii.gz'
    nameInputReferMasksFiles    = '*_lumen.nii.gz'
    nameInputRoiMasksFiles      = '*_lungs.nii.gz'
    prefixPatternInputFiles     = 'vol[0-9][0-9]_*'

    if (args.outpredictmasksdir):
        nameOutputPredictMasksRelPath = args.outpredictmasksdir
    else:
        nameOutputPredictMasksRelPath = nameInputPredictionsRelPath[:-1] + '_Thres%s' % (str(args.thresholdPost))

    nameOutputPredictMasksFiles = lambda in_name: filenamenoextension(in_name).replace('probmap','binmask') + '.nii.gz'
    # ---------- SETTINGS ----------


    workDirsManager        = WorkDirsManager(args.basedir)
    InputPredictionsPath   = workDirsManager.getNameExistPath        (nameInputPredictionsRelPath)
    InputReferMasksPath    = workDirsManager.getNameExistBaseDataPath(nameInputReferMasksRelPath)
    OutputPredictMasksPath = workDirsManager.getNameNewPath          (nameOutputPredictMasksRelPath)

    listInputPredictionsFiles = findFilesDirAndCheck(InputPredictionsPath, nameInputPredictionsFiles)
    listInputReferMasksFiles  = findFilesDirAndCheck(InputReferMasksPath,  nameInputReferMasksFiles)

    if (args.masksToRegionInterest):
        InputRoiMasksPath      = workDirsManager.getNameExistBaseDataPath(nameInputRoiMasksRelPath)
        listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath, nameInputRoiMasksFiles)

        def compute_trachea_masks(in_refermask_array, in_roimask_array):
            return np.where(in_roimask_array == 1, 0, in_refermask_array)



    for i, in_prediction_file in enumerate(listInputPredictionsFiles):
        print("\nInput: \'%s\'..." % (basename(in_prediction_file)))

        in_refermask_file = findFileWithSamePrefix(basename(in_prediction_file), listInputReferMasksFiles,
                                                   prefix_pattern=prefixPatternInputFiles)
        print("Reference mask file: \'%s\'..." % (basename(in_refermask_file)))

        in_prediction_array = FileReader.getImageArray(in_prediction_file)
        in_refermask_array  = FileReader.getImageArray(in_refermask_file)
        print("Predictions of size: %s..." % (str(in_prediction_array.shape)))


        print("Compute prediction masks by Thresholding probability maps with value \'%s\'..." % (args.thresholdPost))
        out_predictmask_array = ThresholdImages.compute(in_prediction_array, args.thresholdPost)

        if (args.masksToRegionInterest):
            print("Attach trachea mask to prediction masks...")

            in_roimask_file = findFileWithSamePrefix(basename(in_prediction_file), listInputRoiMasksFiles,
                                                     prefix_pattern=prefixPatternInputFiles)
            print("RoI mask (lungs) file: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask_array = FileReader.getImageArray(in_roimask_file)
            in_tracheamask_array = compute_trachea_masks(in_refermask_array, in_roimask_array)

            out_predictmask_array = OperationBinaryMasks.join_two_binmasks_one_image(out_predictmask_array, in_tracheamask_array)


        out_file = joinpathnames(OutputPredictMasksPath, nameOutputPredictMasksFiles(basename(in_prediction_file)))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_file), str(out_predictmask_array.shape)))

        FileReader.writeImageArray(out_file, out_predictmask_array)
    #endfor



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('inputpredictiondir', type=str)
    parser.add_argument('outpredictmasksdir', type=str)
    parser.add_argument('--thresholdPost', type=float, default=THRESHOLDPOST)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)