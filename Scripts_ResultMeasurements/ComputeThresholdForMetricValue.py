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
if TYPE_DNNLIBRARY_USED == 'Keras':
    from Networks_Keras.Metrics import *
elif TYPE_DNNLIBRARY_USED == 'Pytorch':
    from Networks_Pytorch.Metrics import *
from Preprocessing.OperationImages import *
from Preprocessing.OperationMasks import *
from tqdm import tqdm
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameInputPredictionsRelPath = args.inputpredictiondir
    nameInputReferMasksRelPath  = 'Airways_Proc/'
    nameInputRoiMasksRelPath    = 'Lungs_Proc/'
    nameInputCentrelinesRelPath = 'Centrelines_Proc/'
    nameInputPredictionsFiles   = '*.nii.gz'
    nameInputReferMasksFiles    = '*_lumen.nii.gz'
    nameInputRoiMasksFiles      = '*_lungs.nii.gz'
    nameInputCentrelinesFiles   = '*_centrelines.nii.gz'
    prefixPatternInputFiles     = 'vol[0-9][0-9]_*'

    metricsEvalThreshold = args.metricsEvalThreshold
    value_metrics_sought = args.ValueMetricsSought
    num_iterEvaluate_max = args.numIterEvaluateMax
    relError_eval_max    = args.relErrorEvalMax
    init_threshold       = args.initThreshold

    ref0_threshold = 0.0
    if (metricsEvalThreshold == 'DiceCoefficient' or
        metricsEvalThreshold == 'AirwayCompleteness'):
        ref0_metrics_value = 0.0
    elif (metricsEvalThreshold == 'AirwayVolumeLeakage'):
        ref0_metrics_value = 1.0
    else:
        message = 'MetricsEvalThreshold \'%s\' not found...' %(metricsEvalThreshold)
        CatchErrorException(message)

    _epsilon = 1.0e-06
    # ---------- SETTINGS ----------


    workDirsManager      = WorkDirsManager(args.basedir)
    InputPredictionsPath = workDirsManager.getNameExistPath(nameInputPredictionsRelPath)
    InputReferMasksPath  = workDirsManager.getNameExistBaseDataPath(nameInputReferMasksRelPath)
    InputCentrelinesPath = workDirsManager.getNameExistBaseDataPath(nameInputCentrelinesRelPath)

    listInputPredictionsFiles = findFilesDirAndCheck(InputPredictionsPath, nameInputPredictionsFiles)
    listInputReferMasksFiles  = findFilesDirAndCheck(InputReferMasksPath,  nameInputReferMasksFiles)
    listInputCentrelinesFiles = findFilesDirAndCheck(InputCentrelinesPath, nameInputCentrelinesFiles)

    if (args.removeTracheaResMetrics):
        InputRoiMasksPath = workDirsManager.getNameExistBaseDataPath(nameInputRoiMasksRelPath)
        listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath, nameInputRoiMasksFiles)

    newgenMetrics = DICTAVAILMETRICFUNS(metricsEvalThreshold)
    fun_EvalThreshold = newgenMetrics.compute_np
    isUseCentrelineRefer = newgenMetrics._use_refer_centreline



    num_validpred_files = len(listInputPredictionsFiles)
    curr_threshold = init_threshold
    old_threshold = ref0_threshold
    old_metrics_value = ref0_metrics_value
    isCompConverged = False


    print("Load all predictions where to evaluate the metrics, total of \'%s\' files..." %(num_validpred_files))
    listin_prediction_array = []
    listin_refermask_array  = []
    with tqdm(total=num_validpred_files) as progressbar:
        for i, in_prediction_file in enumerate(listInputPredictionsFiles):

            in_prediction_array = FileReader.getImageArray(in_prediction_file)

            if (isUseCentrelineRefer):
                in_centreline_file = findFileWithSamePrefix(basename(in_prediction_file).replace('probmap',''), listInputCentrelinesFiles,
                                                            prefix_pattern=prefixPatternInputFiles)
                in_refermask_array = FileReader.getImageArray(in_centreline_file)
            else:
                in_refermask_file = findFileWithSamePrefix(basename(in_prediction_file).replace('probmap',''), listInputReferMasksFiles,
                                                           prefix_pattern=prefixPatternInputFiles)
                in_refermask_array = FileReader.getImageArray(in_refermask_file)

            if (args.removeTracheaResMetrics):
                in_roimask_file = findFileWithSamePrefix(basename(in_prediction_file).replace('probmap',''), listInputRoiMasksFiles,
                                                         prefix_pattern=prefixPatternInputFiles)
                in_roimask_array = FileReader.getImageArray(in_roimask_file)

                in_prediction_array = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(in_prediction_array, in_roimask_array)
                in_refermask_array  = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(in_refermask_array,  in_roimask_array)

            listin_prediction_array.append(in_prediction_array)
            listin_refermask_array.append(in_refermask_array)

            progressbar.update(1)
        #endfor


    for it in range(num_iterEvaluate_max):
        print("Iteration \'%s\'. Evaluate predictions with threshold \'%s\'..." %(it, curr_threshold))

        # Loop over all prediction files and compute the mean metrics over the dataset
        with tqdm(total=num_validpred_files) as progressbar:
            sumrun_metric_value = 0.0
            for i, (in_prediction_array, in_refermask_array) in enumerate(zip(listin_prediction_array, listin_refermask_array)):

                # Threshold probability maps to "curr_threshold"
                in_predictmask_array = ThresholdImages.compute(in_prediction_array, curr_threshold)

                metric_value = fun_EvalThreshold(in_refermask_array, in_predictmask_array)
                sumrun_metric_value += metric_value

                progressbar.update(1)
            #endfor
            curr_metrics_value = sumrun_metric_value / num_validpred_files


        # Compare the mean metrics with the one desired, compute the relative error, and evaluate whether it's close enough
        curr_relError = (curr_metrics_value - value_metrics_sought) / value_metrics_sought

        if abs(curr_relError) < relError_eval_max:
            print("CONVERGED. Found threshold \'%s\', with result metrics \'%s\', rel. error: \'%s\'..." %(curr_threshold, curr_metrics_value,
                                                                                                           abs(curr_relError)))
            isCompConverged = True
            break
        else:
            # Update the threshold following a Newton-Raphson formula
            new_threshold = curr_threshold + (curr_threshold - old_threshold)/(curr_metrics_value - old_metrics_value +_epsilon)*(value_metrics_sought - curr_metrics_value)
            new_threshold = np.clip(new_threshold, 0.0, 1.0)   # clip new value to bounded limits

            print("Not Converged. Result metrics \'%s\', rel. error: \'%s\'. New threshold \'%s\'..." %(curr_metrics_value, curr_relError, new_threshold))
            old_threshold    = curr_threshold
            old_metrics_value= curr_metrics_value
            curr_threshold   = new_threshold
    #endfor

    if not isCompConverged:
        print("ERROR. COMPUTATION NOT CONVERGED...")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('inputpredictiondir', type=str)
    parser.add_argument('--metricsEvalThreshold', type=str, default='AirwayVolumeLeakage')
    parser.add_argument('--ValueMetricsSought', type=float, default=0.12)
    parser.add_argument('--numIterEvaluateMax', type=int, default=20)
    parser.add_argument('--relErrorEvalMax', type=float, default=1.0e-02)
    parser.add_argument('--initThreshold', type=float, default=0.5)
    parser.add_argument('--removeTracheaResMetrics', type=str2bool, default=REMOVETRACHEARESMETRICS)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)