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
from Networks_Pytorch.Metrics import *
from OperationImages.OperationImages import *
from OperationImages.OperationMasks import *
from collections import OrderedDict
import argparse



def main(args):

    workDirsManager       = WorkDirsManager(args.basedir)
    InputPredictMasksPath = workDirsManager.getNameExistPath        (args.inputpredmasksdir)
    InputReferMasksPath   = workDirsManager.getNameExistBaseDataPath(args.nameInputReferMasksRelPath)

    listInputPredictMasksFiles = findFilesDirAndCheck(InputPredictMasksPath)
    listInputReferMasksFiles   = findFilesDirAndCheck(InputReferMasksPath)
    prefixPatternInputFiles    = getFilePrefixPattern(listInputReferMasksFiles[0])

    if (args.removeTracheaCalcMetrics):
        InputCoarseAirwaysPath      = workDirsManager.getNameExistBaseDataPath(args.nameInputCoarseAirwaysRelPath)
        listInputCoarseAirwaysFiles = findFilesDirAndCheck(InputCoarseAirwaysPath)


    listResultMetrics = OrderedDict()
    list_isUse_reference_centrelines = []
    list_isUse_predicted_centrelines = []
    for imetrics in args.listResultMetrics:
        newgen_metrics = DICTAVAILMETRICFUNS(imetrics)
        if newgen_metrics.name_fun_out=='completeness_mod':
            listResultMetrics[newgen_metrics.name_fun_out] = newgen_metrics.compute_fun_np_correct
        else:
            listResultMetrics[newgen_metrics.name_fun_out] = newgen_metrics.compute_np

        list_isUse_reference_centrelines.append(newgen_metrics._isUse_reference_clines)
        list_isUse_predicted_centrelines.append(newgen_metrics._isUse_predicted_clines)
    #endfor
    is_load_refer_cenlines_files = any(list_isUse_reference_centrelines)
    is_load_pred_cenlines_files  = any(list_isUse_predicted_centrelines)

    if (is_load_refer_cenlines_files):
        print("Loading Reference Centrelines...")
        InputReferCentrelinesPath      = workDirsManager.getNameExistBaseDataPath(args.nameInputReferCentrelinesRelPath)
        listInputReferCentrelinesFiles = findFilesDirAndCheck(InputReferCentrelinesPath)

    if (is_load_pred_cenlines_files):
        if not args.inputcenlinesdir:
            message = 'Missing input path for the Predicted Centrelines...'
            CatchErrorException(message)

        print("Loading Predicted Centrelines...")
        InputPredictCentrelinesPath      = workDirsManager.getNameExistPath(args.inputcenlinesdir)
        listInputPredictCentrelinesFiles = findFilesDirAndCheck(InputPredictCentrelinesPath)



    outdict_computedMetrics = OrderedDict()

    for i, in_predictmask_file in enumerate(listInputPredictMasksFiles):
        print("\nInput: \'%s\'..." % (basename(in_predictmask_file)))

        in_refermask_file = findFileWithSamePrefixPattern(basename(in_predictmask_file), listInputReferMasksFiles,
                                                          prefix_pattern=prefixPatternInputFiles)
        print("Reference mask file: \'%s\'..." % (basename(in_refermask_file)))

        in_predictmask_array = FileReader.get_image_array(in_predictmask_file)
        in_refermask_array   = FileReader.get_image_array(in_refermask_file)
        print("Predictions of size: %s..." % (str(in_predictmask_array.shape)))

        if (is_load_refer_cenlines_files):
            in_refercenline_file = findFileWithSamePrefixPattern(basename(in_predictmask_file), listInputReferCentrelinesFiles,
                                                                 prefix_pattern=prefixPatternInputFiles)
            print("Reference centrelines file: \'%s\'..." % (basename(in_refercenline_file)))
            in_refercenline_array = FileReader.get_image_array(in_refercenline_file)

        if (is_load_pred_cenlines_files):
            in_predictcenline_file = findFileWithSamePrefixPattern(basename(in_predictmask_file), listInputPredictCentrelinesFiles,
                                                                   prefix_pattern=prefixPatternInputFiles)
            print("Predicted centrelines file: \'%s\'..." % (basename(in_predictcenline_file)))
            in_predictcenline_array = FileReader.get_image_array(in_predictcenline_file)


        if (args.removeTracheaCalcMetrics):
            print("Remove trachea and main bronchii masks in computed metrics...")
            in_coarseairways_file = findFileWithSamePrefixPattern(basename(in_predictmask_file), listInputCoarseAirwaysFiles,
                                                                  prefix_pattern=prefixPatternInputFiles)
            print("Coarse Airways mask file: \'%s\'..." % (basename(in_coarseairways_file)))

            in_coarseairways_array = FileReader.get_image_array(in_coarseairways_file)

            print("Dilate coarse airways masks 4 levels to remove completely trachea and main bronchi from ground-truth...")
            in_coarseairways_array = MorphoDilateMasks.compute(in_coarseairways_array, num_iters=4)

            in_predictmask_array = OperationMasks.substract_two_masks(in_predictmask_array, in_coarseairways_array)
            in_refermask_array   = OperationMasks.substract_two_masks(in_refermask_array,   in_coarseairways_array)

            if (is_load_refer_cenlines_files):
                in_refercenline_array = OperationMasks.substract_two_masks(in_refercenline_array, in_coarseairways_array)

            if (is_load_pred_cenlines_files):
                in_predictcenline_array = OperationMasks.substract_two_masks(in_predictcenline_array, in_coarseairways_array)


        # *******************************************************************************
        # Compute and store Metrics
        print("\nCompute the Metrics:")
        key_casename = getSubstringPatternFilename(basename(in_predictmask_file), substr_pattern=prefixPatternInputFiles)[:-1]
        outdict_computedMetrics[key_casename] = []

        for j, (imetr_name, imetrics_fun) in enumerate(listResultMetrics.items()):
            if list_isUse_reference_centrelines[j]:
                in_referdata_array = in_refercenline_array
            else:
                in_referdata_array = in_refermask_array
            if list_isUse_predicted_centrelines[j]:
                in_predictdata_array = in_predictcenline_array
            else:
                in_predictdata_array = in_predictmask_array

            try:
            #'try' statement to 'catch' issues when predictions are 'null' (for extreme threshold values)
                if imetr_name=='completeness_mod':
                    outres_metrics = imetrics_fun(in_referdata_array, in_predictdata_array, in_refercenline_array)
                else:
                    outres_metrics = imetrics_fun(in_referdata_array, in_predictdata_array)
            except:  # set dummy value for cases with issues
                outres_metrics = -1.0
            print("\'%s\': %s..." % (imetr_name, outres_metrics))

            outdict_computedMetrics[key_casename].append(outres_metrics)
        # endfor
    # endfor


    # write out computed metrics in file
    out_results_filename = joinpathnames(args.inputpredmasksdir, args.outputfile)
    fout = open(out_results_filename, 'w')

    strheader = ', '.join(['/case/'] + ['/%s/'%(key) for key in listResultMetrics.keys()]) + '\n'
    fout.write(strheader)

    for (in_casename, outlist_computedMetrics) in outdict_computedMetrics.items():
        list_outdata = [in_casename] + ['%0.6f'%(elem) for elem in outlist_computedMetrics]
        strdata = ', '.join(list_outdata) + '\n'
        fout.write(strdata)
    #endfor
    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('inputpredmasksdir', type=str)
    parser.add_argument('--inputcenlinesdir', type=str, default=None)
    parser.add_argument('--outputfile', type=str, default='result_metrics.txt')
    parser.add_argument('--listResultMetrics', type=parseListarg, default=LISTRESULTMETRICS)
    parser.add_argument('--removeTracheaCalcMetrics', type=str2bool, default=REMOVETRACHEACALCMETRICS)
    parser.add_argument('--nameInputReferMasksRelPath', type=str, default=NAME_RAWLABELS_RELPATH)
    parser.add_argument('--nameInputCoarseAirwaysRelPath', type=str, default=NAME_RAWCOARSEAIRWAYS_RELPATH)
    parser.add_argument('--nameInputReferCentrelinesRelPath', type=str, default=NAME_RAWCENTRELINES_RELPATH)
    args = parser.parse_args()

    if not args.inputcenlinesdir:
        args.inputcenlinesdir = args.inputpredmasksdir

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)
