#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.Constants import *
from CommonUtil.ErrorMessages import *
from CommonUtil.FileReaders import *
from CommonUtil.FunctionsUtil import *
from CommonUtil.PlotsManager import *
from CommonUtil.WorkDirsManager import *
from Networks.Metrics import *
from Preprocessing.OperationsImages import *
from collections import OrderedDict
import argparse


def main(args):

    # ---------- SETTINGS ----------
    nameInputMasksRelPath   = 'ProcMasks'
    nameTracheaMasksRelPath = 'ProcAllMasks'

    # Get the file list:
    namePredictionsFiles  = 'predict_probmaps*.nii.gz'
    nameInputMasksFiles   = '*lumen.nii.gz'
    nameTracheaMasksFiles = '*lumen_traquea.nii.gz'

    # template search files
    tempSearchInputFiles = 'av[0-9]*'

    if (args.calcMasksThresholding):
        suffixPostProcessThreshold = '_thres%s'%(str(args.thresholdValue).replace('.','-'))
        if (args.attachTracheaToCalcMasks):
            suffixPostProcessThreshold += '_withtrachea'
    else:
        suffixPostProcessThreshold = ''

    # create file to save accuracy measures on test sets
    nameAccuracyPredictFiles = 'predict_accuracy_tests%s.txt'%(suffixPostProcessThreshold)

    def update_name_outfile(in_name, in_acc):
        pattern_accval = getExtractSubstringPattern(in_name, 'acc[0-9]*')
        new_accval_int = np.round(100*in_acc)
        out_name = filenamenoextension(in_name).replace('predict_probmaps','predict_binmasks').replace(pattern_accval, 'acc%2.0f'%(new_accval_int))
        return out_name + '%s.nii.gz'%(suffixPostProcessThreshold)

    tempOutPredictMasksFilename = update_name_outfile
    # ---------- SETTINGS ----------


    workDirsManager       = WorkDirsManager(args.basedir)
    BaseDataPath          = workDirsManager.getNameBaseDataPath()
    InputPredictDataPath  = workDirsManager.getNameExistPath(args.basedir, args.predictionsdir)
    InputMasksPath        = workDirsManager.getNameExistPath(BaseDataPath, nameInputMasksRelPath)
    OutputPredictMasksPath= workDirsManager.getNameNewPath  (args.basedir, args.predictionsdir)

    listPredictionsFiles    = findFilesDir(InputPredictDataPath, namePredictionsFiles)
    listGrndTruthMasksFiles = findFilesDir(InputMasksPath,       nameInputMasksFiles)

    nbPredictionsFiles    = len(listPredictionsFiles)
    nbGrndTruthMasksFiles = len(listGrndTruthMasksFiles)

    # Run checkers
    if (nbPredictionsFiles == 0):
        message = "0 Predictions found in dir \'%s\'" %(InputPredictDataPath)
        CatchErrorException(message)
    if (nbGrndTruthMasksFiles == 0):
        message = "0 Ground-truth Masks found in dir \'%s\'" %(InputMasksPath)
        CatchErrorException(message)


    if (args.calcMasksThresholding and args.attachTracheaToCalcMasks):

        TracheaMasksPath = workDirsManager.getNameExistPath(BaseDataPath, nameTracheaMasksRelPath)

        listTracheaMasksFiles = findFilesDir(TracheaMasksPath, nameTracheaMasksFiles)

        nbTracheaMasksFiles = len(listTracheaMasksFiles)

        if (nbGrndTruthMasksFiles != nbTracheaMasksFiles):
            message = "num Ground-truth Masks %i not equal to num Trachea Masks %i" %(nbGrndTruthMasksFiles, nbTracheaMasksFiles)
            CatchErrorException(message)


    computePredictAccuracy = DICTAVAILMETRICFUNS(args.predictAccuracyMetrics, use_in_Keras=False)

    listFuns_Metrics = {imetrics: DICTAVAILMETRICFUNS(imetrics, use_in_Keras=False) for imetrics in args.listPostprocessMetrics}
    out_predictAccuracyFilename = joinpathnames(InputPredictDataPath, nameAccuracyPredictFiles)
    fout = open(out_predictAccuracyFilename, 'w')

    strheader = '/case/ ' + ' '.join(['/%s/' % (key) for (key, _) in listFuns_Metrics.iteritems()]) + '\n'
    fout.write(strheader)



    for i, predict_probmaps_file in enumerate(listPredictionsFiles):

        print('\'%s\'...' %(predict_probmaps_file))

        name_prefix_case = getExtractSubstringPattern(basename(predict_probmaps_file),
                                                      tempSearchInputFiles)

        for iterfile in listGrndTruthMasksFiles:
            if name_prefix_case in iterfile:
                grndtruth_masks_file = iterfile
        #endfor
        print("assigned to '%s'..." %(basename(grndtruth_masks_file)))


        predict_probmaps_array = FileReader.getImageArray(predict_probmaps_file)
        grndtruth_masks_array  = FileReader.getImageArray(grndtruth_masks_file)

        print("Predictions masks array of size: %s..." % (str(predict_probmaps_array.shape)))


        if (args.calcMasksThresholding):
            print("Threshold probability maps to compute binary masks with threshold value %s..." % (args.thresholdValue))

            predict_masks_array = ThresholdImages.compute(predict_probmaps_array, args.thresholdValue)

            if (args.attachTracheaToCalcMasks):
                print("IMPORTANT: Attach masks of Trachea to computed prediction masks...")

                for iterfile in listTracheaMasksFiles:
                    if name_prefix_case in iterfile:
                        trachea_masks_file = iterfile
                # endfor
                print("assigned to: '%s'..." % (basename(trachea_masks_file)))

                trachea_masks_array = FileReader.getImageArray(trachea_masks_file)

                predict_masks_array   = OperationsBinaryMasks.join_two_binmasks_one_image(predict_masks_array,   trachea_masks_array)
                grndtruth_masks_array = OperationsBinaryMasks.join_two_binmasks_one_image(grndtruth_masks_array, trachea_masks_array)



        accuracy = computePredictAccuracy(grndtruth_masks_array, predict_masks_array)

        list_predictAccuracy = OrderedDict()

        for (key, value) in listFuns_Metrics.iteritems():
            acc_value = value(grndtruth_masks_array, predict_masks_array)
            list_predictAccuracy[key] = acc_value
        # endfor


        # print list accuracies on screen
        for (key, value) in list_predictAccuracy.iteritems():
            print("Computed '%s': %s..." %(key, value))
        #endfor

        # print list accuracies in file
        strdata = '\'%s\''%(name_prefix_case) + ' ' + ' '.join([str(value) for (_,value) in list_predictAccuracy.iteritems()]) +'\n'
        fout.write(strdata)


        # Save thresholded final prediction masks
        if (args.calcMasksThresholding and args.saveThresholdImages):
            print("Saving prediction thresholded binary masks, with dims: %s..." %(tuple2str(predict_masks_array.shape)))

            out_predictMasksFilename = joinpathnames(OutputPredictMasksPath, tempOutPredictMasksFilename(predict_probmaps_file, accuracy))

            FileReader.writeImageArray(out_predictMasksFilename, predict_masks_array)
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