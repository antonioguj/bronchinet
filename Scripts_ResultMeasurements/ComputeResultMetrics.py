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
from collections import OrderedDict
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameInputPredictMasksRelPath = args.inputpredictmasksdir
    nameInputReferMasksRelPath   = 'Airways_Proc/'
    nameInputRoiMasksRelPath     = 'Lungs_Proc/'
    nameInputCentrelinesRelPath  = 'Centrelines_Proc/'
    nameInputPredictMasksFiles   = 'binmask_*.nii.gz'
    nameInputReferMasksFiles     = '*_lumen.nii.gz'
    nameInputRoiMasksFiles       = '*_lungs.nii.gz'
    nameInputCentrelinesFiles    = '*_centrelines.nii.gz'
    prefixPatternInputFiles      = 'vol[0-9][0-9]_*'

    if (args.removeTracheaResMetrics):
        nameResultMetricsFile = 'result_metrics_notraquea.txt'
    else:
        nameResultMetricsFile = 'result_metrics_withtraquea.txt'
    # ---------- SETTINGS ----------


    workDirsManager       = WorkDirsManager(args.basedir)
    InputPredictMasksPath = workDirsManager.getNameExistPath        (nameInputPredictMasksRelPath)
    InputReferMasksPath   = workDirsManager.getNameExistBaseDataPath(nameInputReferMasksRelPath)
    InputCentrelinesPath  = workDirsManager.getNameExistBaseDataPath(nameInputCentrelinesRelPath)

    listInputPredictMasksFiles = findFilesDirAndCheck(InputPredictMasksPath, nameInputPredictMasksFiles)
    listInputReferMasksFiles   = findFilesDirAndCheck(InputReferMasksPath,   nameInputReferMasksFiles)
    listInputCentrelinesFiles  = findFilesDirAndCheck(InputCentrelinesPath,  nameInputCentrelinesFiles)

    if (args.removeTracheaResMetrics):
        InputRoiMasksPath      = workDirsManager.getNameExistBaseDataPath(nameInputRoiMasksRelPath)
        listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath, nameInputRoiMasksFiles)


    listResultMetrics = OrderedDict()
    list_isUse_centrelines = []
    for imetrics in args.listResultMetrics:
        listResultMetrics[imetrics] = DICTAVAILMETRICFUNS(imetrics).compute_np_safememory
        list_isUse_centrelines.append(DICTAVAILMETRICFUNS(imetrics)._use_refer_centreline)
    #endfor
    isUseCentrelineFiles = any(list_isUse_centrelines)

    out_resultmetrics_filename = joinpathnames(nameInputPredictMasksRelPath, nameResultMetricsFile)
    fout = open(out_resultmetrics_filename, 'w')
    strheader = '/case/ ' + ' '.join(['/%s/' % (key) for (key, _) in listResultMetrics.iteritems()]) + '\n'
    fout.write(strheader)



    for i, in_predictmask_file in enumerate(listInputPredictMasksFiles):
        print("\nInput: \'%s\'..." % (basename(in_predictmask_file)))

        in_refermask_file = findFileWithSamePrefix(basename(in_predictmask_file).replace('probmap',''), listInputReferMasksFiles,
                                                   prefix_pattern=prefixPatternInputFiles)
        print("Reference mask file: \'%s\'..." % (basename(in_refermask_file)))

        in_predictmask_array = FileReader.getImageArray(in_predictmask_file)
        in_refermask_array   = FileReader.getImageArray(in_refermask_file)
        print("Predictions of size: %s..." % (str(in_predictmask_array.shape)))


        if (isUseCentrelineFiles):
            in_centreline_file = findFileWithSamePrefix(basename(in_predictmask_file).replace('probmap',''), listInputCentrelinesFiles,
                                                        prefix_pattern=prefixPatternInputFiles)
            print("Centrelines file: \'%s\'..." % (basename(in_centreline_file)))
            in_centrelines_array = FileReader.getImageArray(in_centreline_file)


        if (args.removeTracheaResMetrics):
            print("Remove trachea mask in result measurements...")

            in_roimask_file = findFileWithSamePrefix(basename(in_predictmask_file).replace('probmap',''), listInputRoiMasksFiles,
                                                     prefix_pattern=prefixPatternInputFiles)
            print("RoI mask (lungs) file: \'%s\'..." % (basename(in_roimask_file)))

            in_roimask_array = FileReader.getImageArray(in_roimask_file)

            in_predictmask_array = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(in_predictmask_array, in_roimask_array)
            in_refermask_array   = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(in_refermask_array,   in_roimask_array)

            if (isUseCentrelineFiles):
                in_centrelines_array = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(in_centrelines_array, in_roimask_array)


        # Compute Result Metrics
        list_result_metrics = OrderedDict()
        for i, (key, value) in enumerate(listResultMetrics.iteritems()):
            if list_isUse_centrelines[i]:
                metric_value = value(in_centrelines_array, in_predictmask_array)
            else:
                metric_value = value(in_refermask_array, in_predictmask_array)
            list_result_metrics[key] = metric_value
        # endfor

        # print list metircs on screen and in file
        prefix_casename = getSubstringPatternFilename(basename(in_predictmask_file), substr_pattern=prefixPatternInputFiles)[:-1]
        strdata = '\'%s\'' % (prefix_casename)
        for (key, value) in list_result_metrics.iteritems():
            print("Metric \'%s\': %s..." %(key, value))
            strdata += ' %s'%(str(value))
        #endfor
        strdata += '\n'
        fout.write(strdata)
    #endfor

    #close list accuracies file
    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('inputpredictmasksdir')
    parser.add_argument('--listResultMetrics', type=parseListarg, default=LISTRESULTMETRICS)
    parser.add_argument('--removeTracheaResMetrics', type=str2bool, default=REMOVETRACHEARESMETRICS)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
