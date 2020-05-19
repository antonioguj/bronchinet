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
    outfilename_metrics_eachcase = 'results_metrics_diffThres_%s.csv'
    outfilename_metrics_meanall  = 'meanAllcases_results_metrics_diffThres.txt'

    # parameters to draw ROC curve
    inlist_thresholds = [0.0]
    # num_thresholds = 9
    # range_threshold = [-10, -2]
    # inlist_thresholds += (np.logspace(range_threshold[0], range_threshold[1], num_thresholds)).tolist()
    # num_thresholds = 9
    # range_threshold = [0.1, 0.9]
    # inlist_thresholds += (np.linspace(range_threshold[0], range_threshold[1], num_thresholds)).tolist()
    # num_thresholds = 9
    # range_threshold = [-2, -10]
    # inlist_thresholds += [1.0 - elem for elem in (np.logspace(range_threshold[0], range_threshold[1], num_thresholds)).tolist()]
    # #allowedDistance = 0
    inlist_thresholds += [1.0]
    print("List of Threshold values: %s" % (inlist_thresholds))
    # ---------- SETTINGS ----------


    workDirsManager     = WorkDirsManager(args.basedir)
    InputPosteriorsPath = workDirsManager.getNameExistPath        (args.inputpredictionsdir)
    InputReferMasksPath = workDirsManager.getNameExistBaseDataPath(args.nameInputReferMasksRelPath)
    OutputFilesPath     = workDirsManager.getNameNewPath          (args.outputdir)

    listInputPosteriorsFiles = findFilesDirAndCheck(InputPosteriorsPath)
    listInputReferMasksFiles = findFilesDirAndCheck(InputReferMasksPath)
    prefixPatternInputFiles  = getFilePrefixPattern(listInputReferMasksFiles[0])

    if (args.removeTracheaCalcMetrics):
        InputCoarseAirwaysPath      = workDirsManager.getNameExistBaseDataPath(args.nameInputCoarseAirwaysRelPath)
        listInputCoarseAirwaysFiles = findFilesDirAndCheck(InputCoarseAirwaysPath)
        #InputRoiMasksPath      = workDirsManager.getNameExistBaseDataPath(args.nameInputRoiMasksRelPath)
        #listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath)


    listMetricsROCcurve = OrderedDict()
    list_isUse_reference_centrelines = []
    list_isUse_predicted_centrelines = []
    for imetrics in args.listMetricsROCcurve:
        newgen_metrics = DICTAVAILMETRICFUNS(imetrics)
        listMetricsROCcurve[newgen_metrics.name_fun_out] = newgen_metrics.compute_np

        list_isUse_reference_centrelines.append(newgen_metrics._isUse_reference_clines)
        list_isUse_predicted_centrelines.append(newgen_metrics._isUse_predicted_clines)
    #endfor
    is_load_refer_cenlines_files = any(list_isUse_reference_centrelines)
    is_load_pred_cenlines_files  = any(list_isUse_predicted_centrelines)

    if (is_load_refer_cenlines_files):
        print("Loading Reference Centrelines...")
        InputReferCentrelinesPath      = workDirsManager.getNameExistBaseDataPath(args.nameInputReferCentrelinesRelPath)
        listInputReferCentrelinesFiles = findFilesDirAndCheck(InputReferCentrelinesPath)


    num_input_files  = len(listInputPosteriorsFiles)
    num_comp_metrics = len(listMetricsROCcurve)
    num_thresholds   = len(inlist_thresholds)
    out_computedMetrics_allcases = np.zeros((num_input_files, num_thresholds, num_comp_metrics))


    for i, in_posterior_file in enumerate(listInputPosteriorsFiles):
        print("\nInput: \'%s\'..." % (basename(in_posterior_file)))

        in_refermask_file = findFileWithSamePrefixPattern(basename(in_posterior_file), listInputReferMasksFiles,
                                                          prefix_pattern=prefixPatternInputFiles)
        print("Reference mask file: \'%s\'..." % (basename(in_refermask_file)))

        in_posterior_array = FileReader.get_image_array(in_posterior_file)
        in_refermask_array = FileReader.get_image_array(in_refermask_file)
        print("Predictions of size: %s..." % (str(in_posterior_array.shape)))

        if (is_load_refer_cenlines_files):
            in_refercenline_file = findFileWithSamePrefixPattern(basename(in_posterior_file), listInputReferCentrelinesFiles,
                                                                 prefix_pattern=prefixPatternInputFiles)
            print("Reference centrelines file: \'%s\'..." % (basename(in_refercenline_file)))
            in_refercenline_array = FileReader.get_image_array(in_refercenline_file)


        if (args.removeTracheaCalcMetrics):
            print("Remove trachea and main bronchii masks in computed metrics...")

            in_coarseairways_file = findFileWithSamePrefixPattern(basename(in_posterior_file), listInputCoarseAirwaysFiles,
                                                                  prefix_pattern=prefixPatternInputFiles)
            print("Coarse Airways mask file: \'%s\'..." % (basename(in_coarseairways_file)))

            in_coarseairways_array = FileReader.get_image_array(in_coarseairways_file)

            print("Dilate coarse airways masks 4 levels to remove completely trachea and main bronchi from ground-truth...")
            in_coarseairways_array = MorphoDilateMasks.compute(in_coarseairways_array, num_iters=4)

            in_posterior_array = OperationMasks.substract_two_masks(in_posterior_array, in_coarseairways_array)
            in_refermask_array = OperationMasks.substract_two_masks(in_refermask_array, in_coarseairways_array)

            if (is_load_refer_cenlines_files):
                in_refercenline_array = OperationMasks.substract_two_masks(in_refercenline_array, in_coarseairways_array)


        # *******************************************************************************
        # Compute and store Metrics at all thresholds
        print("Compute Metrics with Predictions computed with all thresholds \'%s\'..." %(inlist_thresholds))
        key_casename = getSubstringPatternFilename(basename(in_posterior_file), substr_pattern=prefixPatternInputFiles)[:-1]
        outdict_computedMetrics = OrderedDict()

        with tqdm(total=num_thresholds) as progressbar:
            for j, in_thres_value in enumerate(inlist_thresholds):

                # Compute the binary masks by thresholding
                in_predictmask_array = ThresholdImages.compute(in_posterior_array, in_thres_value)

                if (is_load_pred_cenlines_files):
                    # Compute centrelines by thinning the binary masks
                    try:
                        #'try' statement to 'catch' issues when predictions are 'weird' (for extreme threshold values)
                        in_predictcenline_array = ThinningMasks.compute(in_predictmask_array)
                    except:
                        in_predictcenline_array = np.zeros_like(in_predictmask_array)


                outdict_computedMetrics[in_thres_value] = []
                for k, (imetr_name, imetrics_fun) in enumerate(listMetricsROCcurve.iteritems()):
                    if list_isUse_reference_centrelines[k]:
                        in_referdata_array = in_refercenline_array
                    else:
                        in_referdata_array = in_refermask_array
                    if list_isUse_predicted_centrelines[k]:
                        in_predictdata_array = in_predictcenline_array
                    else:
                        in_predictdata_array = in_predictmask_array

                    try:
                        #'try' statement to 'catch' issues when predictions are 'null' (for extreme threshold values)
                        outres_metrics = imetrics_fun(in_referdata_array, in_predictdata_array)
                    except:
                        # set dummy value for cases with issues
                        outres_metrics = -1.0

                    outdict_computedMetrics[in_thres_value].append(outres_metrics)
                # endfor

                out_computedMetrics_allcases[i,j,:] = outdict_computedMetrics[in_thres_value]
                progressbar.update(1)
            # endfor
        # *******************************************************************************


        # write out computed metrics in file
        out_results_filename = outfilename_metrics_eachcase %(key_casename)
        out_results_filename = joinpathnames(OutputFilesPath, out_results_filename)
        if isExistfile(out_results_filename):
            fout = open(out_results_filename, 'a')
        else:
            fout = open(out_results_filename, 'w')
            strheader = ', '.join(['/thres/'] + ['/%s/'%(key) for key in listMetricsROCcurve.keys()]) + '\n'
            fout.write(strheader)

        for (in_thres, outlist_computedMetrics) in outdict_computedMetrics.iteritems():
            list_outdata = ['%0.3f'%(in_thres)] + ['%0.6f'%(elem) for elem in outlist_computedMetrics]
            strdata = ', '.join(list_outdata) + '\n'
            fout.write(strdata)
        # endfor
        fout.close()
    # endfor



    # Compute global metrics as mean over all files
    outmeanallcases_computedMetrics = np.mean(out_computedMetrics_allcases, axis=0)

    out_filename = joinpathnames(OutputFilesPath, outfilename_metrics_meanall)
    if isExistfile(out_filename):
        fout = open(out_filename, 'a')
    else:
        fout = open(out_filename, 'w')
        strheader = ', '.join(['/thres/'] + ['/%s/'%(key) for key in listMetricsROCcurve.keys()]) + '\n'
        fout.write(strheader)

    for i, in_thres in enumerate(inlist_thresholds):
        list_outdata = ['%0.3f'%(in_thres)] + ['%0.6f'%(elem) for elem in outmeanallcases_computedMetrics[i]]
        strdata = ', '.join(list_outdata) + '\n'
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
