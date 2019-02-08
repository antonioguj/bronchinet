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
from Networks_Keras.Metrics import *
from Preprocessing.OperationImages import *
from Preprocessing.OperationMasks import *
from collections import OrderedDict
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameInputPredictionsRelPath = args.predictionsdirs
    nameInputReferMasksRelPath = 'Airways_Full'
    nameInputRoiMasksRelPath = 'Lungs_Full'
    nameOutputPredictionsRelPath = nameInputPredictionsRelPath

    nameInputPredictionsFiles = 'predict_probmaps*.nii.gz'
    nameInputReferMasksFiles = '*_lumen.nii.gz'
    nameInputRoiMasksFiles = '*_lungs.nii.gz'
    # prefixPatternInputFiles = 'av[0-9][0-9]*'

    if (args.calcMasksThresholding):
        suffixPostProcessThreshold = '_thres%s'%(str(args.thresholdValue).replace('.','-'))
        if (args.attachTracheaToCalcMasks):
            suffixPostProcessThreshold += '_withtrachea'
    else:
        suffixPostProcessThreshold = ''

    nameAccuracyPredictFiles = 'predict_accuracy_tests%s.txt'%(suffixPostProcessThreshold)

    def nameOutputFiles(in_name, in_acc):
        out_name = filenamenoextension(in_name).replace('predict_probmaps','predict_binmasks') + '_acc%2.0f' %(np.round(100*in_acc))
        return out_name + '%s.nii.gz'%(suffixPostProcessThreshold)
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(args.basedir)
    BaseDataPath = workDirsManager.getNameBaseDataPath()
    InputPredictionsPath = workDirsManager.getNameExistPath(args.basedir, nameInputPredictionsRelPath)
    InputReferMasksPath = workDirsManager.getNameExistPath(BaseDataPath, nameInputReferMasksRelPath)
    OutputPredictionsPath = workDirsManager.getNameNewPath(args.basedir, nameOutputPredictionsRelPath)

    listInputPredictionsFiles = findFilesDirAndCheck(InputPredictionsPath, nameInputPredictionsFiles)
    listInputReferMasksFiles = findFilesDirAndCheck(InputReferMasksPath, nameInputReferMasksFiles)

    if (args.calcMasksThresholding and args.attachTracheaToCalcMasks):
        InputRoiMasksPath = workDirsManager.getNameExistPath(BaseDataPath, nameInputRoiMasksRelPath)

        listInputRoiMasksFiles = findFilesDirAndCheck(InputRoiMasksPath, nameInputRoiMasksFiles)

        def compute_trachea_masks(refermask_array, roimask_array):
            return np.where(roimask_array == 1, 0, refermask_array)


    computePredictAccuracy = DICTAVAILMETRICFUNS(args.predictAccuracyMetrics).compute_np_safememory

    listFuns_Metrics = {imetrics: DICTAVAILMETRICFUNS(imetrics).compute_np_safememory for imetrics in args.listPostprocessMetrics}
    out_predictAccuracyFilename = joinpathnames(InputPredictionsPath, nameAccuracyPredictFiles)
    fout = open(out_predictAccuracyFilename, 'w')

    strheader = '/case/ ' + ' '.join(['/%s/' % (key) for (key, _) in listFuns_Metrics.iteritems()]) + '\n'
    fout.write(strheader)



    for i, in_prediction_file in enumerate(listInputPredictionsFiles):
        print("\nInput: \'%s\'..." % (basename(in_prediction_file)))

        in_refermask_file = findFileWithSamePrefix(basename(in_prediction_file), listInputReferMasksFiles)
        print("Refer mask file: \'%s\'..." % (basename(in_refermask_file)))
        in_roimask_file = findFileWithSamePrefix(basename(in_prediction_file), listInputRoiMasksFiles)
        print("RoI mask (lungs) file: \'%s\'..." % (basename(in_roimask_file)))

        prediction_array = FileReader.getImageArray(in_prediction_file)
        refermask_array = FileReader.getImageArray(in_refermask_file)
        roimask_array = FileReader.getImageArray(in_roimask_file)

        print("Predictions of size: %s..." % (str(prediction_array.shape)))

        if (args.calcMasksThresholding):
            print("Compute prediction masks by thresholding probability maps to value %s..." % (args.thresholdValue))
            prediction_array = ThresholdImages.compute(prediction_array, args.thresholdValue)


        if (args.attachTracheaToCalcMasks):
            print("Attach trachea mask to computed prediction masks...")
            trachea_masks_array = compute_trachea_masks(refermask_array, roimask_array)

            prediction_array = OperationBinaryMasks.join_two_binmasks_one_image(prediction_array, trachea_masks_array)
        else:
            refermask_array = OperationBinaryMasks.apply_mask_exclude_voxels_fillzero(refermask_array, roimask_array)


        accuracy = computePredictAccuracy(refermask_array, prediction_array)

        list_predictAccuracy = OrderedDict()
        for (key, value) in listFuns_Metrics.iteritems():
            acc_value = value(refermask_array, prediction_array)
            list_predictAccuracy[key] = acc_value
        # endfor

        # print list accuracies on screen
        for (key, value) in list_predictAccuracy.iteritems():
            print("Computed '%s': %s..." %(key, value))
        #endfor

        # print list accuracies in file
        prefix_casename = basename(in_prediction_file).split('_')[0]
        strdata = '\'%s\''%(prefix_casename) + ' ' + ' '.join([str(value) for (_,value) in list_predictAccuracy.iteritems()]) +'\n'
        fout.write(strdata)


        out_file = joinpathnames(OutputPredictionsPath, nameOutputFiles(basename(in_prediction_file, accuracy)))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_file), str(prediction_array.shape)))

        FileReader.writeImageArray(out_file, prediction_array)
    #endfor

    #close list accuracies file
    fout.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--predictionsdir', default='Predictions_NEW')
    parser.add_argument('--predictAccuracyMetrics', default=PREDICTACCURACYMETRICS)
    parser.add_argument('--listPostprocessMetrics', type=parseListarg, default=LISTPOSTPROCESSMETRICS)
    parser.add_argument('--calcMasksThresholding', type=str2bool, default=CALCMASKSTHRESHOLDING)
    parser.add_argument('--thresholdValue', type=float, default=THRESHOLDVALUE)
    parser.add_argument('--attachTracheaToCalcMasks', type=str2bool, default=ATTACHTRAQUEATOCALCMASKS)
    parser.add_argument('--saveThresholdImages', type=str2bool, default=SAVETHRESHOLDIMAGES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)