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
from Preprocessing.OperationImages import *
from collections import OrderedDict
import argparse
np.random.seed(2017)



def main(args):
    # ---------- SETTINGS ----------
    nameInputPredictionsRelPath = args.inputpredictiondir
    nameInputReferMasksRelPath  = 'Airways_Proc/'
    nameInputRoiMasksRelPath    = 'Lungs_Proc/'
    nameInputCentrelinesRelPath = 'Centrelines_Full'
    nameInputPredictionsFiles   = 'probmap_*.nii.gz'
    nameInputReferMasksFiles    = '*_lumen.nii.gz'
    nameInputRoiMasksFiles      = '*_lungs.nii.gz'
    nameInputCentrelinesFiles   = '*_centrelines.nii.gz'
    prefixPatternInputFiles     = 'vol[0-9][0-9]_*'
    nameOutROCmetricsFile       = 'dataROC_metrics_%s.txt'
    nameOutMeanROCmetricsFile   = 'dataROC_metrics_mean.txt'

    if args.outputdir:
        nameOutputFilesRelPath = args.outputdir
    else:
        nameOutputFilesRelPath = nameInputPredictionsRelPath

    # parameters to draw ROC curve
    #um_thresholds = 9
    #range_threshold = [0.1, 0.9]
    #list_thresholds = (np.linspace(range_threshold[0], range_threshold[1], num_thresholds)).tolist()
    num_thresholds = 5
    range_threshold = [-10, -6]
    list_thresholds = (np.logspace(range_threshold[0], range_threshold[1], num_thresholds)).tolist()
    #list_thresholds = [1.0 - elem for elem in reversed(list_thresholds)]
    #allowedDistance = 0
    # ---------- SETTINGS ----------


    workDirsManager       = WorkDirsManager(args.basedir)
    InputPredictionsPath = workDirsManager.getNameExistPath        (args.inputpredictiondir)
    InputReferMasksPath   = workDirsManager.getNameExistBaseDataPath(nameInputReferMasksRelPath)
    InputCentrelinesPath  = workDirsManager.getNameExistBaseDataPath(nameInputCentrelinesRelPath)
    OutputFilesPath       = workDirsManager.getNameNewPath          (args.outputdir)

    listInputPredictionsFiles = findFilesDirAndCheck(InputPredictionsPath, nameInputPredictionsFiles)
    listInputReferMasksFiles   = findFilesDirAndCheck(InputReferMasksPath,   nameInputReferMasksFiles)
    listInputCentrelinesFiles  = findFilesDirAndCheck(InputCentrelinesPath,  nameInputCentrelinesFiles)

    if (args.removeTracheaResMetrics):
        InputRoiMasksPath      = workDirsManager.getNameExistBaseDataPath(nameInputRoiMasksRelPath)
        listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath, nameInputRoiMasksFiles)


    listMetricsROCcurve = OrderedDict()
    list_isUse_centrelines = []
    for imetrics in args.listMetricsROCcurve:
        newgenMetrics = DICTAVAILMETRICFUNS(imetrics)
        listMetricsROCcurve[imetrics] = newgenMetrics.compute_np_safememory
        list_isUse_centrelines.append(newgenMetrics._use_refer_centreline)
    #endfor
    isUseCentrelineFiles = any(list_isUse_centrelines)

    print("List of Threshold values: %s" % (list_thresholds))

    nbInputFiles = len(listInputPredictionsFiles)
    nbCompMetrics = len(listMetricsROCcurve)
    list_computed_metrics = np.zeros((nbInputFiles, num_thresholds, nbCompMetrics))



    for i, in_prediction_file in enumerate(listInputPredictionsFiles):
        print("\nInput: \'%s\'..." % (basename(in_prediction_file)))

        in_refermask_file = findFileWithSamePrefix(basename(in_predictmask_file).replace('probmap',''), listInputReferMasksFiles,
                                                   prefix_pattern=prefixPatternInputFiles)
        print("Reference mask file: \'%s\'..." % (basename(in_refermask_file)))

        in_prediction_array = FileReader.getImageArray(in_prediction_file)
        in_refermask_array  = FileReader.getImageArray(in_refermask_file)
        print("Predictions of size: %s..." % (str(in_prediction_array.shape)))

        if (isUseCentrelineFiles):
            in_centreline_file = findFileWithSamePrefix(basename(in_prediction_file).replace('probmap',''), listInputCentrelinesFiles,
                                                        prefix_pattern=prefixPatternInputFiles)
            print("Centrelines file: \'%s\'..." % (basename(in_centreline_file)))
            in_centreline_array = FileReader.getImageArray(in_centreline_file)


        if (args.removeTracheaResMetrics):
            print("Remove trachea and main bronchi masks in computed metrics...")

            in_roimask_file = findFileWithSamePrefix(basename(in_predictmask_file).replace('probmap',''), listInputRoiMasksFiles,
                                                     prefix_pattern=prefixPatternInputFiles)
            print("RoI mask (lungs) file: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask_array = FileReader.getImageArray(in_roimask_file)

            in_prediction_array = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(in_prediction_array, in_roimask_array)
            in_refermask_array  = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(in_refermask_array,  in_roimask_array)

            if (isUseCentrelineFiles):
                in_centreline_array = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(in_centreline_array, in_roimask_array)


        print("Compute and store metrics at all thresholds...",)
        for j, threshold_value in enumerate(list_thresholds):
            print("Compute metrics at threshold %s..." %(threshold_value))

            in_predictmask_array = ThresholdImages.compute(in_prediction_array, threshold_value)

            for k, (key, value) in enumerate(listMetricsROCcurve.iteritems()):
                if list_isUse_centrelines[k]:
                    metric_value = value(in_centreline_array, in_predictmask_array)
                else:
                    metric_value = value(in_refermask_array, in_predictmask_array)

                print("Metric \'%s\': %s..." % (key, metric_value))
                list_computed_metrics[i,j,k] = metric_value
            # endfor
        #endfor
        print("...Done")


        # write out computed metrics in file
        out_filename = joinpathnames(OutputDataPath, nameOutROCmetricsFile%(basename_file))
        if isExistfile(out_filename):
            fout = open(out_filename, 'a')
        else:
            fout = open(out_filename, 'w')
            strheader = '/threshold/ ' + ' '.join(['/%s/' % (key) for (key, _) in listMetricsROCcurve.iteritems()]) + '\n'
            fout.write(strheader)

        for j in range(num_thresholds):
            strdata = '%s %s\n' % (list_thresholds[j], ' '.join([str(elem) for elem in list_computed_metrics[i,j]]))
            fout.write(strdata)
        # endfor
        fout.close()
    #endfor


    # Compute global metrics as mean over all files
    list_mean_computed_metrics = np.mean(list_computed_metrics, axis=0)

    out_filename = joinpathnames(OutputDataPath, nameOutMeanROCmetricsFile)
    if isExistfile(out_filename):
        fout = open(out_filename, 'a')
    else:
        fout = open(out_filename, 'w')
        strheader = '/threshold/ ' + ' '.join(['/%s/' % (key) for (key, _) in listMetricsROCcurve.iteritems()]) + '\n'
        fout.write(strheader)

    for i in range(num_thresholds):
        strdata = '%s %s\n' % (list_thresholds[i], ' '.join([str(elem) for elem in list_mean_computed_metrics[i]]))
        fout.write(strdata)
    # endfor
    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('inputpredictions', type=str)
    parser.add_argument('--outputdir', type=str)
    parser.add_argument('--listMetricsROCcurve', type=parseListarg, default=LISTMETRICSROCCURVE)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
