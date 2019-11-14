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
#if TYPE_DNNLIBRARY_USED == 'Keras':
#    from Networks_Keras.Metrics import *
#elif TYPE_DNNLIBRARY_USED == 'Pytorch':
#    from Networks_Pytorch.Metrics import *
from Networks_Pytorch.Metrics import *
from Preprocessing.OperationImages import *
from Preprocessing.OperationMasks import *
from collections import OrderedDict
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameInputPredictMasksRelPath      = args.inputpredictmasksdir
    nameInputReferMasksRelPath        = 'Airways_Proc/'
    nameInputRoiMasksRelPath          = 'Lungs_Proc/'
    nameInputPredictCentrelinesRelPath= args.inputcentrelinesdir
    nameInputReferCentrelinesRelPath  = 'Centrelines_Proc/'
    nameInputPredictMasksFiles        = '*_binmask.nii.gz'
    nameInputReferMasksFiles          = '*_lumen.nii.gz'
    nameInputRoiMasksFiles            = '*_lungs.nii.gz'
    nameInputPredictCentrelinesFiles  = '*_cenlines.nii.gz'
    nameInputReferCentrelinesFiles    = '*_cenlines.nii.gz'
    prefixPatternInputFiles           = 'vol[0-9][0-9]_*'

    if (args.removeTracheaResMetrics):
        nameOutResultMetricsFile = 'result_metrics_notrachea.txt'
    else:
        nameOutResultMetricsFile = 'result_metrics_withtrachea.txt'
    # ---------- SETTINGS ----------


    workDirsManager       = WorkDirsManager(args.basedir)
    InputPredictMasksPath = workDirsManager.getNameExistPath        (nameInputPredictMasksRelPath)
    InputReferMasksPath   = workDirsManager.getNameExistBaseDataPath(nameInputReferMasksRelPath)

    listInputPredictMasksFiles = findFilesDirAndCheck(InputPredictMasksPath)
    listInputReferMasksFiles   = findFilesDirAndCheck(InputReferMasksPath)

    if (args.removeTracheaResMetrics):
        InputRoiMasksPath      = workDirsManager.getNameExistBaseDataPath(nameInputRoiMasksRelPath)
        listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath)


    listResultMetrics = OrderedDict()
    list_isUse_reference_centrelines = []
    list_isUse_predicted_centrelines = []
    for imetrics in args.listResultMetrics:
        newgenMetrics = DICTAVAILMETRICFUNS(imetrics)
        if newgenMetrics.name_fun_out=='completeness_mod':
            listResultMetrics[newgenMetrics.name_fun_out] = newgenMetrics.compute_fun_np_correct
        else:
            listResultMetrics[newgenMetrics.name_fun_out] = newgenMetrics.compute_np

        list_isUse_reference_centrelines.append(newgenMetrics._isUse_reference_clines)
        list_isUse_predicted_centrelines.append(newgenMetrics._isUse_predicted_clines)
    #endfor
    isLoadReferenceCentrelineFiles = any(list_isUse_reference_centrelines)
    isLoadPredictedCentrelineFiles = any(list_isUse_predicted_centrelines)

    if (isLoadReferenceCentrelineFiles):
        print("Loading Reference Centrelines...")
        InputReferCentrelinesPath      = workDirsManager.getNameExistBaseDataPath(nameInputReferCentrelinesRelPath)
        listInputReferCentrelinesFiles = findFilesDirAndCheck(InputReferCentrelinesPath, nameInputReferCentrelinesFiles)

    if (isLoadPredictedCentrelineFiles):
        if not nameInputPredictCentrelinesRelPath:
            message = 'Missing input path for the predicted centrelines...'
            CatchErrorException(message)

        print("Loading Predicted Centrelines...")
        InputPredictCentrelinesPath      = workDirsManager.getNameExistPath(nameInputPredictCentrelinesRelPath)
        listInputPredictCentrelinesFiles = findFilesDirAndCheck(InputPredictCentrelinesPath)


    nbInputFiles  = len(listInputPredictMasksFiles)
    nbCompMetrics = len(listResultMetrics)
    list_casenames = []
    list_computed_metrics = np.zeros((nbInputFiles, nbCompMetrics))



    for i, in_predictmask_file in enumerate(listInputPredictMasksFiles):
        print("\nInput: \'%s\'..." % (basename(in_predictmask_file)))

        in_refermask_file = findFileWithSamePrefix(basename(in_predictmask_file), listInputReferMasksFiles,
                                                   prefix_pattern=prefixPatternInputFiles)
        print("Reference mask file: \'%s\'..." % (basename(in_refermask_file)))

        in_predictmask_array = FileReader.getImageArray(in_predictmask_file)
        in_refermask_array   = FileReader.getImageArray(in_refermask_file)
        print("Predictions of size: %s..." % (str(in_predictmask_array.shape)))

        if (isLoadReferenceCentrelineFiles):
            in_refercenline_file = findFileWithSamePrefix(basename(in_predictmask_file), listInputReferCentrelinesFiles,
                                                          prefix_pattern=prefixPatternInputFiles)
            print("Reference centrelines file: \'%s\'..." % (basename(in_refercenline_file)))
            in_refercenline_array = FileReader.getImageArray(in_refercenline_file)

        if (isLoadPredictedCentrelineFiles):
            in_predictcenline_file = findFileWithSamePrefix(basename(in_predictmask_file), listInputPredictCentrelinesFiles,
                                                            prefix_pattern=prefixPatternInputFiles)
            print("Predicted centrelines file: \'%s\'..." % (basename(in_predictcenline_file)))
            in_predictcenline_array = FileReader.getImageArray(in_predictcenline_file)


        if (args.removeTracheaResMetrics):
            print("Remove trachea and main bronchi masks in computed metrics...")

            in_roimask_file = findFileWithSamePrefix(basename(in_predictmask_file), listInputRoiMasksFiles,
                                                     prefix_pattern=prefixPatternInputFiles)
            print("RoI mask (lungs) file: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask_array = FileReader.getImageArray(in_roimask_file)

            in_predictmask_array = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(in_predictmask_array, in_roimask_array)
            in_refermask_array   = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(in_refermask_array,   in_roimask_array)

            if (isLoadReferenceCentrelineFiles):
                in_refercenline_array = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(in_refercenline_array, in_roimask_array)

            if (isLoadPredictedCentrelineFiles):
                in_predictcenline_array = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(in_predictcenline_array, in_roimask_array)


        # Compute and store Metrics
        for j, (key, value) in enumerate(listResultMetrics.iteritems()):
            if list_isUse_reference_centrelines[j]:
                in_referYdata_array = in_refercenline_array
            else:
                in_referYdata_array = in_refermask_array
            if list_isUse_predicted_centrelines[j]:
                in_predictYdata_array = in_predictcenline_array
            else:
                in_predictYdata_array = in_predictmask_array

            try:
            # 'try' statement to 'catch' issues when predictions are 'null' (for extreme threshold values)
                if key=='completeness_mod':
                    metric_value = value(in_referYdata_array, in_predictYdata_array, in_refercenline_array)
                else:
                    metric_value = value(in_referYdata_array, in_predictYdata_array)
            except:  # set dummy value for cases with issues
                metric_value = -1.0
            print("Metric \'%s\': %s..." % (key, metric_value))
            list_computed_metrics[i,j] = metric_value
        # endfor

        prefix_casename = getSubstringPatternFilename(basename(in_predictmask_file), substr_pattern=prefixPatternInputFiles)[:-1]
        list_casenames.append(prefix_casename)
    #endfor

    # write out computed metrics in file
    out_resultmetrics_filename = joinpathnames(nameInputPredictMasksRelPath, nameOutResultMetricsFile)
    fout = open(out_resultmetrics_filename, 'w')
    strheader = '/case/, ' + ', '.join(['/%s/' % (key) for (key, _) in listResultMetrics.iteritems()]) + '\n'
    fout.write(strheader)

    for i in range(nbInputFiles):
        strdata = '\'%s\', %s\n' % (list_casenames[i], ', '.join([str(elem) for elem in list_computed_metrics[i]]))
        fout.write(strdata)
    #endfor
    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASEDIR)
    parser.add_argument('inputpredictmasksdir', type=str)
    parser.add_argument('--inputcentrelinesdir', type=str, default=None)
    parser.add_argument('--listResultMetrics', type=parseListarg, default=LISTRESULTMETRICS)
    parser.add_argument('--removeTracheaResMetrics', type=str2bool, default=REMOVETRACHEARESMETRICS)
    args = parser.parse_args()

    if not args.inputcentrelinesdir:
        args.inputcentrelinesdir = args.inputpredictmasksdir

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)