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
from OperationImages.OperationImages import *
from OperationImages.OperationMasks import *
from tqdm import tqdm
import argparse



def main(args):
    # ---------- SETTINGS ----------
    metricsEvalThreshold = args.metricsEvalThreshold
    metrics_value_sought = args.metricsValueSought
    num_iterEvaluate_max = args.numIterEvaluateMax
    relError_eval_max    = args.relErrorEvalMax
    init_threshold_value = args.initThresholdValue

    ref0_threshold_value = 0.0
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


    workDirsManager     = WorkDirsManager(args.basedir)
    InputPosteriorsPath = workDirsManager.getNameExistPath        (args.inputposteriorsdir)
    InputReferMasksPath = workDirsManager.getNameExistBaseDataPath(args.nameInputReferMasksRelPath)

    listInputPosteriorsFiles = findFilesDirAndCheck(InputPosteriorsPath)
    listInputReferMasksFiles = findFilesDirAndCheck(InputReferMasksPath)
    prefixPatternInputFiles  = getFilePrefixPattern(listInputReferMasksFiles[0])

    if (args.removeTracheaCalcMetrics):
        InputCoarseAirwaysPath      = workDirsManager.getNameExistBaseDataPath(args.nameInputCoarseAirwaysRelPath)
        listInputCoarseAirwaysFiles = findFilesDirAndCheck(InputCoarseAirwaysPath)
        

    newgen_metrics = DICTAVAILMETRICFUNS(metricsEvalThreshold)
    fun_metrics_evalThreshold = newgen_metrics.compute_np
    is_load_refer_cenlines_files = newgen_metrics._isUse_reference_clines
    is_load_pred_cenlines_files  = newgen_metrics._isUse_predicted_clines

    if (is_load_refer_cenlines_files):
        print("Loading Reference Centrelines...")
        InputReferCentrelinesPath      = workDirsManager.getNameExistBaseDataPath(args.nameInputReferCentrelinesRelPath)
        listInputReferCentrelinesFiles = findFilesDirAndCheck(InputReferCentrelinesPath)


    num_validpred_files = len(listInputPosteriorsFiles)
    curr_thres_value  = init_threshold_value
    old_thres_value   = ref0_threshold_value
    old_metrics_value = ref0_metrics_value
    isCompConverged   = False



    print("Load all predictions where to evaluate the metrics, total of \'%s\' files..." %(num_validpred_files))
    listin_posterior_array = []
    listin_reference_array = []
    listin_coarseairways_array = []
    with tqdm(total=num_validpred_files) as progressbar:
        for i, in_posterior_file in enumerate(listInputPosteriorsFiles):

            in_posterior_array = FileReader.get_image_array(in_posterior_file)
            listin_posterior_array.append(in_posterior_array)

            if (is_load_refer_cenlines_files):
                in_refercenline_file = findFileWithSamePrefixPattern(basename(in_posterior_file), listInputReferCentrelinesFiles,
                                                                     prefix_pattern=prefixPatternInputFiles)
                in_refercenline_array = FileReader.get_image_array(in_refercenline_file)
                in_reference_array = in_refercenline_array
            else:
                in_refermask_file = findFileWithSamePrefixPattern(basename(in_posterior_file), listInputReferMasksFiles,
                                                                  prefix_pattern=prefixPatternInputFiles)
                in_refermask_array = FileReader.get_image_array(in_refermask_file)
                in_reference_array = in_refermask_array

            if (args.removeTracheaResMetrics):
                in_coarseairways_file = findFileWithSamePrefixPattern(basename(in_posterior_file), listInputCoarseAirwaysFiles,
                                                                      prefix_pattern=prefixPatternInputFiles)
                in_coarseairways_array = FileReader.get_image_array(in_coarseairways_file)

                in_coarseairways_array = MorphoDilateMasks.compute(in_coarseairways_array, num_iters=4)
                listin_coarseairways_array.append(in_coarseairways_array)

                in_reference_array = OperationMasks.substract_two_masks(in_reference_array, in_coarseairways_array)

            listin_reference_array.append(in_reference_array)
            progressbar.update(1)
        # endfor
    # *******************************************************************************


    for iter in range(num_iterEvaluate_max):
        print("Iteration \'%s\'. Evaluate predictions with threshold \'%s\'..." %(iter, curr_thres_value))

        # Loop over all prediction files and compute the mean metrics over the dataset
        with tqdm(total=num_validpred_files) as progressbar:
            sumrun_res_metrics = 0.0
            for i, (in_posterior_array, in_referdata_array) in enumerate(zip(listin_posterior_array, listin_reference_array)):

                # Compute the binary masks by thresholding the posteriors
                in_predictmask_array = ThresholdImages.compute(in_posterior_array, curr_thres_value)

                if (args.isconnectedmasks):
                    # Compute the first connected component from the binary masks
                    in_predictmask_array = FirstConnectedRegionMasks.compute(in_predictmask_array, connectivity_dim=1)

                if (is_load_pred_cenlines_files):
                    # Compute centrelines by thinning the binary masks
                    try:
                        #'try' statement to 'catch' issues when predictions are 'weird' (for extreme threshold values)
                        in_predictcenline_array = ThinningMasks.compute(in_predictmask_array)
                    except:
                        in_predictcenline_array = np.zeros_like(in_predictmask_array)
                    in_predictdata_array = in_predictcenline_array
                else:
                    in_predictdata_array = in_predictmask_array

                if (args.removeTracheaCalcMetrics):
                    # Remove the trachea and main bronchii from the binary masks
                    in_predictdata_array = OperationMasks.substract_two_masks(in_predictdata_array, in_coarseairways_array)
                # **********************************************


                outres_metrics = fun_metrics_evalThreshold(in_referdata_array, in_predictdata_array)
                sumrun_res_metrics += outres_metrics

                progressbar.update(1)
            #endfor
            curr_res_metrics = sumrun_res_metrics / num_validpred_files


        # Compare the mean metrics with the one desired, compute the relative error, and evaluate whether it's close enough
        curr_relError = (curr_res_metrics - metrics_value_sought) / metrics_value_sought

        if abs(curr_relError) < relError_eval_max:
            print("CONVERGED. Found threshold \'%s\', with result metrics \'%s\', rel. error: \'%s\'..." %(curr_thres_value, curr_res_metrics, abs(curr_relError)))
            isCompConverged = True
            break
        else:
            # Update the threshold following a Newton-Raphson formula
            new_threshold = curr_thres_value + (curr_thres_value - old_thres_value)/(curr_res_metrics - old_metrics_value +_epsilon)*(metrics_value_sought - curr_res_metrics)
            new_threshold = np.clip(new_threshold, 0.0, 1.0)   # clip new value to bounded limits

            print("Not Converged. Result metrics \'%s\', rel. error: \'%s\'. New threshold \'%s\'..." %(curr_res_metrics, curr_relError, new_threshold))
            old_thres_value  = curr_thres_value
            old_metrics_value= curr_res_metrics
            curr_thres_value = new_threshold
    #endfor

    if not isCompConverged:
        print("ERROR. COMPUTATION NOT CONVERGED...")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('inputposteriorsdir', type=str)
    parser.add_argument('--metricsEvalThreshold', type=str, default='AirwayVolumeLeakage')
    parser.add_argument('--metricsValueSought', type=float, default=0.13)
    parser.add_argument('--removeTracheaResMetrics', type=str2bool, default=REMOVETRACHEACALCMETRICS)
    parser.add_argument('--isconnectedmasks', type=str2bool, default=False)
    parser.add_argument('--nameInputReferMasksRelPath', type=str, default=NAME_RAWLABELS_RELPATH)
    parser.add_argument('--nameInputCoarseAirwaysRelPath', type=str, default=NAME_RAWCOARSEAIRWAYS_RELPATH)
    parser.add_argument('--nameInputReferCentrelinesRelPath', type=str, default=NAME_RAWCENTRELINES_RELPATH)
    parser.add_argument('--numIterEvaluateMax', type=int, default=20)
    parser.add_argument('--relErrorEvalMax', type=float, default=1.0e-04)
    parser.add_argument('--initThresholdValue', type=float, default=0.5)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)