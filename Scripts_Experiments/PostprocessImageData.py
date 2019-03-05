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
    nameInputPredictionsRelPath = args.predictionsdir
    nameInputReferMasksRelPath = 'Airways_Full'
    nameInputRoiMasksRelPath = 'Lungs_Full'
    nameInputCentrelinesRelPath = 'Centrelines_Full'
    nameOutputPredictionsRelPath = nameInputPredictionsRelPath

    nameInputPredictionsFiles = 'predict-probmaps_*.nii.gz'
    nameInputReferMasksFiles = '*_lumen.nii.gz'
    nameInputRoiMasksFiles = '*_lungs.nii.gz'
    nameInputCentrelinesFiles = '*_centrelines.nii.gz'
    # prefixPatternInputFiles = 'av[0-9][0-9]*'

    if (args.calcMasksThresholding):
        suffixPostProcessThreshold = '_thres%s'%(str(args.thresholdValue).replace('.','-'))
        if (args.attachTracheaToCalcMasks):
            suffixPostProcessThreshold += '_withtrachea'
        else:
            suffixPostProcessThreshold += '_notrachea'
    else:
        suffixPostProcessThreshold = ''

    nameAccuracyPredictFiles = 'predict_accuracy_tests%s.txt'%(suffixPostProcessThreshold)

    def nameOutputFiles(in_name, in_acc):
        out_name = filenamenoextension(in_name).replace('predict-probmaps','predict-binmasks') + '_acc%2.0f' %(np.round(100*in_acc))
        return out_name + '%s.nii.gz'%(suffixPostProcessThreshold)
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(args.basedir)
    BaseDataPath = workDirsManager.getNameBaseDataPath()
    InputPredictionsPath = workDirsManager.getNameExistPath(args.basedir, nameInputPredictionsRelPath)
    InputReferenceMasksPath = workDirsManager.getNameExistPath(BaseDataPath, nameInputReferMasksRelPath)
    OutputPredictionsPath = workDirsManager.getNameNewPath(args.basedir, nameOutputPredictionsRelPath)

    listInputPredictionsFiles = findFilesDirAndCheck(InputPredictionsPath, nameInputPredictionsFiles)
    listInputReferenceMasksFiles = findFilesDirAndCheck(InputReferenceMasksPath, nameInputReferMasksFiles)

    if (args.masksToRegionInterest):
        InputRoiMasksPath = workDirsManager.getNameExistPath(BaseDataPath, nameInputRoiMasksRelPath)
        listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath, nameInputRoiMasksFiles)

        if (args.attachTracheaToCalcMasks):
            def compute_trachea_masks(refermask_array, roimask_array):
                return np.where(roimask_array == 1, 0, refermask_array)


    listPostProcessMetrics = OrderedDict()
    list_isUseCenlineFiles = []
    for imetrics in args.listPostprocessMetrics:
        listPostProcessMetrics[imetrics] = DICTAVAILMETRICFUNS(imetrics).compute_np_safememory
        list_isUseCenlineFiles.append(DICTAVAILMETRICFUNS(imetrics)._is_cenline_grndtru)
    #endfor
    isuse_centreline_files = any(list_isUseCenlineFiles)

    if isuse_centreline_files:
        InputCentrelinesPath = workDirsManager.getNameExistPath(BaseDataPath, nameInputCentrelinesRelPath)
        listInputCentrelinesFiles = findFilesDirAndCheck(InputCentrelinesPath, nameInputCentrelinesFiles)


    out_predictAccuracyFilename = joinpathnames(InputPredictionsPath, nameAccuracyPredictFiles)
    fout = open(out_predictAccuracyFilename, 'w')
    strheader = '/case/ ' + ' '.join(['/%s/' % (key) for (key, _) in listPostProcessMetrics.iteritems()]) + '\n'
    fout.write(strheader)



    for i, in_prediction_file in enumerate(listInputPredictionsFiles):
        print("\nInput: \'%s\'..." % (basename(in_prediction_file)))

        in_refermask_file = findFileWithSamePrefix(basename(in_prediction_file).replace('predict-probmaps',''),
                                                   listInputReferenceMasksFiles,
                                                   prefix_pattern='vol[0-9][0-9]_')
        print("Refer mask file: \'%s\'..." % (basename(in_refermask_file)))

        prediction_array = FileReader.getImageArray(in_prediction_file)
        refermask_array = FileReader.getImageArray(in_refermask_file)
        print("Predictions of size: %s..." % (str(prediction_array.shape)))


        if (args.calcMasksThresholding):
            print("Compute prediction masks by thresholding probability maps to value %s..." % (args.thresholdValue))
            prediction_array = ThresholdImages.compute(prediction_array, args.thresholdValue)


        if isuse_centreline_files:
            in_centreline_file = findFileWithSamePrefix(basename(in_prediction_file).replace('predict-probmaps', ''),
                                                        listInputCentrelinesFiles,
                                                        prefix_pattern='vol[0-9][0-9]_')
            print("Centrelines file: \'%s\'..." % (basename(in_centreline_file)))
            centrelines_array = FileReader.getImageArray(in_centreline_file)


        if (args.masksToRegionInterest):
            in_roimask_file = findFileWithSamePrefix(basename(in_prediction_file).replace('predict-probmaps',''),
                                                     listInputRoiMasksFiles,
                                                     prefix_pattern='vol[0-9][0-9]_')
            print("RoI mask (lungs) file: \'%s\'..." % (basename(in_roimask_file)))
            roimask_array = FileReader.getImageArray(in_roimask_file)

            if (args.attachTracheaToCalcMasks):
                print("Attach trachea mask to computed prediction masks...")
                trachea_masks_array = compute_trachea_masks(refermask_array, roimask_array)
                prediction_array = OperationBinaryMasks.join_two_binmasks_one_image(prediction_array,
                                                                                    trachea_masks_array)
            else:
                prediction_array = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(prediction_array,
                                                                                           roimask_array)
                refermask_array = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(refermask_array,
                                                                                          roimask_array)
                if isuse_centreline_files:
                    centrelines_array = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(centrelines_array,
                                                                                                roimask_array)


        # ---------- COMPUTE POST PROCESSING MEASURES ----------
        list_postprocess_measures = OrderedDict()
        for i, (key, value) in enumerate(listPostProcessMetrics.iteritems()):
            if list_isUseCenlineFiles[i]:
                acc_value = value(centrelines_array, prediction_array)
            else:
                acc_value = value(refermask_array, prediction_array)
            list_postprocess_measures[key] = acc_value
        # endfor
        main_postprocess_accuracy = list_postprocess_measures.values()[0]


        # print list accuracies on screen and in file
        prefix_casename = getSubstringPatternFilename(basename(in_prediction_file), substr_pattern='vol[0-9][0-9]_')[:-1]
        strdata = '\'%s\'' % (prefix_casename)
        for (key, value) in list_postprocess_measures.iteritems():
            print("Metric \'%s\': %s..." %(key, value))
            strdata += ' %s'%(str(value))
        #endfor
        strdata += '\n'
        fout.write(strdata)
        # ---------- COMPUTE POST PROCESSING MEASURES ----------


        out_file = joinpathnames(OutputPredictionsPath, nameOutputFiles(basename(in_prediction_file), main_postprocess_accuracy))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_file), str(prediction_array.shape)))

        FileReader.writeImageArray(out_file, prediction_array)
    #endfor

    #close list accuracies file
    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--predictionsdir', default='Predictions_NEW')
    parser.add_argument('--listPostprocessMetrics', type=parseListarg, default=LISTPOSTPROCESSMETRICS)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--calcMasksThresholding', type=str2bool, default=True)
    parser.add_argument('--thresholdValue', type=float, default=THRESHOLDVALUE)
    parser.add_argument('--attachTracheaToCalcMasks', type=str2bool, default=ATTACHTRAQUEATOCALCMASKS)
    parser.add_argument('--saveThresholdImages', type=str2bool, default=SAVETHRESHOLDIMAGES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
