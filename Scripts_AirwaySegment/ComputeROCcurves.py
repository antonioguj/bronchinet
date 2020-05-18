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
from PlotsManager.FrocUtil import computeFROC, computeROC_Completeness_VolumeLeakage
from OperationImages.OperationImages import *
from OperationImages.OperationMasks import *
from collections import OrderedDict
from tqdm import tqdm
import argparse
np.random.seed(2017)



def main(args):
    # ---------- SETTINGS ----------
    nameOutROCmetricsFile     = 'dataROC_metrics_%s.txt'
    nameOutMeanROCmetricsFile = 'mean_dataROC_metrics.txt'

    # parameters to draw ROC curve
    list_thresholds = [0.0]
    # num_thresholds = 9
    # range_threshold = [-10, -2]
    # list_thresholds += (np.logspace(range_threshold[0], range_threshold[1], num_thresholds)).tolist()
    # num_thresholds = 9
    # range_threshold = [0.1, 0.9]
    # list_thresholds += (np.linspace(range_threshold[0], range_threshold[1], num_thresholds)).tolist()
    # num_thresholds = 9
    # range_threshold = [-2, -10]
    # list_thresholds += [1.0 - elem for elem in (np.logspace(range_threshold[0], range_threshold[1], num_thresholds)).tolist()]
    # #allowedDistance = 0
    list_thresholds += [1.0]
    print("List of Threshold values: %s" % (list_thresholds))
    # ---------- SETTINGS ----------


    workDirsManager      = WorkDirsManager(args.basedir)
    InputPredictionsPath = workDirsManager.getNameExistPath        (args.inputpredictionsdir)
    InputReferMasksPath  = workDirsManager.getNameExistBaseDataPath(args.nameInputReferMasksRelPath)
    OutputFilesPath      = workDirsManager.getNameNewPath          (args.outputdir)

    listInputPredictionsFiles = findFilesDirAndCheck(InputPredictionsPath)
    listInputReferMasksFiles  = findFilesDirAndCheck(InputReferMasksPath)
    prefixPatternInputFiles   = getFilePrefixPattern(listInputReferMasksFiles[0])

    if (args.removeTracheaCalcMetrics):
        InputRoiMasksPath      = workDirsManager.getNameExistBaseDataPath(args.nameInputRoiMasksRelPath)
        listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath)


    listMetricsROCcurve = OrderedDict()
    list_isUse_reference_centrelines = []
    list_isUse_predicted_centrelines = []
    for imetrics in args.listMetricsROCcurve:
        newgenMetrics = DICTAVAILMETRICFUNS(imetrics)
        listMetricsROCcurve[newgenMetrics.name_fun_out] = newgenMetrics.compute_np

        list_isUse_reference_centrelines.append(newgenMetrics._isUse_reference_clines)
        list_isUse_predicted_centrelines.append(newgenMetrics._isUse_predicted_clines)
    #endfor
    isLoadReferenceCentrelineFiles = any(list_isUse_reference_centrelines)
    isLoadPredictedCentrelineFiles = any(list_isUse_predicted_centrelines)

    if (isLoadReferenceCentrelineFiles):
        print("Loading Reference Centrelines...")
        InputReferCentrelinesPath      = workDirsManager.getNameExistBaseDataPath(args.nameInputReferCentrelinesRelPath)
        listInputReferCentrelinesFiles = findFilesDirAndCheck(InputReferCentrelinesPath)


    nbInputFiles = len(listInputPredictionsFiles)
    nbCompMetrics = len(listMetricsROCcurve)
    num_thresholds = len(list_thresholds)
    list_computed_metrics = np.zeros((nbInputFiles, num_thresholds, nbCompMetrics))



    for i, in_prediction_file in enumerate(listInputPredictionsFiles):
        print("\nInput: \'%s\'..." % (basename(in_prediction_file)))

        in_refermask_file = findFileWithSamePrefixPattern(basename(in_prediction_file), listInputReferMasksFiles,
                                                          prefix_pattern=prefixPatternInputFiles)
        print("Reference mask file: \'%s\'..." % (basename(in_refermask_file)))

        in_prediction_array = FileReader.get_image_array(in_prediction_file)
        in_refermask_array  = FileReader.get_image_array(in_refermask_file)
        print("Predictions of size: %s..." % (str(in_prediction_array.shape)))

        if (isLoadReferenceCentrelineFiles):
            in_refercenline_file = findFileWithSamePrefixPattern(basename(in_prediction_file), listInputReferCentrelinesFiles,
                                                                 prefix_pattern=prefixPatternInputFiles)
            print("Reference centrelines file: \'%s\'..." % (basename(in_refercenline_file)))
            in_refercenline_array = FileReader.get_image_array(in_refercenline_file)


        if (args.removeTracheaCalcMetrics):
            print("Remove trachea and main bronchi masks in computed metrics...")

            in_roimask_file = findFileWithSamePrefixPattern(basename(in_prediction_file), listInputRoiMasksFiles,
                                                            prefix_pattern=prefixPatternInputFiles)
            print("RoI mask (lungs) file: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask_array = FileReader.get_image_array(in_roimask_file)

            in_prediction_array = OperationMasks.mask_exclude_regions_fillzero(in_prediction_array, in_roimask_array)
            in_refermask_array  = OperationMasks.mask_exclude_regions_fillzero(in_refermask_array, in_roimask_array)

            if (isLoadReferenceCentrelineFiles):
                in_refercenline_array = OperationMasks.mask_exclude_regions_fillzero(in_refercenline_array, in_roimask_array)


        print("Compute Metrics at all thresholds \'%s\'..." %(list_thresholds))
        with tqdm(total=num_thresholds) as progressbar:
            for j, threshold_value in enumerate(list_thresholds):

                # Compute the masks by thresholding
                in_predictmask_array = ThresholdImages.compute(in_prediction_array, threshold_value)

                if (isLoadPredictedCentrelineFiles):
                    # Compute centrelines by thinning the masks
                    try:
                        # 'try' statement to 'catch' issues when predictions are 'weird' (for extreme threshold values)
                        in_predictcenline_array = ThinningMasks.compute(in_predictmask_array)
                    except:
                        in_predictcenline_array = np.zeros_like(in_predictmask_array)


                for k, (key, value) in enumerate(listMetricsROCcurve.iteritems()):
                    if list_isUse_reference_centrelines[k]:
                        in_referYdata_array = in_refercenline_array
                    else:
                        in_referYdata_array = in_refermask_array
                    if list_isUse_predicted_centrelines[k]:
                        in_predictYdata_array = in_predictcenline_array
                    else:
                        in_predictYdata_array = in_predictmask_array

                    try:
                        # 'try' statement to 'catch' issues when predictions are 'null' (for extreme threshold values)
                        metric_value = value(in_referYdata_array, in_predictYdata_array)
                    except:
                        # set dummy value for cases with issues
                        metric_value = -1.0
                    list_computed_metrics[i,j,k] = metric_value
                # endfor

                progressbar.update(1)
            #endfor


        # write out computed metrics in file
        prefix_casename = getSubstringPatternFilename(basename(in_prediction_file), substr_pattern=prefixPatternInputFiles)[:-1]
        out_filename = joinpathnames(OutputFilesPath, nameOutROCmetricsFile%(prefix_casename))

        if isExistfile(out_filename):
            fout = open(out_filename, 'a')
        else:
            fout = open(out_filename, 'w')
            strheader = ', '.join(['/thres/'] + ['/%s/'%(key) for (key, _) in listMetricsROCcurve.iteritems()]) + '\n'
            fout.write(strheader)

        for j in range(num_thresholds):
            strdata = ', '.join(['%0.3f'%(list_thresholds[j])] + ['%0.6f'%(elem) for elem in list_computed_metrics[i]]) + '\n'
            fout.write(strdata)
        # endfor
        fout.close()
    #endfor


    # Compute global metrics as mean over all files
    list_mean_computed_metrics = np.mean(list_computed_metrics, axis=0)

    out_filename = joinpathnames(OutputFilesPath, nameOutMeanROCmetricsFile)
    if isExistfile(out_filename):
        fout = open(out_filename, 'a')
    else:
        fout = open(out_filename, 'w')
        strheader = ', '.join(['/thres/'] + ['/%s/'%(key) for (key, _) in listMetricsROCcurve.iteritems()]) + '\n'
        fout.write(strheader)

    for i in range(num_thresholds):
        strdata = ', '.join(['%0.3f'%(list_thresholds[j])] + ['%0.6f' % (elem) for elem in list_mean_computed_metrics[i]]) + '\n'
        fout.write(strdata)
    # endfor
    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('inputpredictionsdir', type=str)
    parser.add_argument('--outputdir', type=str, default=None)
    parser.add_argument('--nameInputReferMasksRelPath', type=str, default=NAME_RAWLABELS_RELPATH)
    parser.add_argument('--nameInputRoiMasksRelPath', type=str, default=NAME_RAWROIMASKS_RELPATH)
    parser.add_argument('--nameInputReferCentrelinesRelPath', type=str, default=NAME_RAWCENTRELINES_RELPATH)
    parser.add_argument('--listMetricsROCcurve', type=parseListarg, default=LISTMETRICSROCCURVE)
    parser.add_argument('--removeTracheaCalcMetrics', type=str2bool, default=REMOVETRACHEACALCMETRICS)
    args = parser.parse_args()

    if not args.outputdir:
        nameOutputFilesRelPath = args.inputpredictionsdir

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)