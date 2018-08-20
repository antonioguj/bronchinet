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
    nameOriginMasksRelPath = 'ProcMasks'

    # Get the file list:
    namePredictionsFiles = 'predict_probmaps*.nii.gz'
    nameOriginMasksFiles = '*.nii.gz'

    # template search files
    tempSearchInputFiles = 'av[0-9]*'

    if (args.thresholdOutProbMaps):
        suffixPostProcessThreshold = '_thres%s'%(str(args.thresholdValue).replace('.','-'))
    else:
        suffixPostProcessThreshold = ''

    # create file to save accuracy measures on test sets
    nameAccuracyPredictFiles = 'predict_accuracy_tests%s.txt'%(suffixPostProcessThreshold)

    tempOutPredictMasksFilename = lambda in_name: filenamenoextension(in_name).replace('predict_probmaps','predict_binmasks') + '%s.nii.gz'%(suffixPostProcessThreshold)
    # ---------- SETTINGS ----------


    workDirsManager  = WorkDirsManager(args.basedir)
    BaseDataPath     = workDirsManager.getNameBaseDataPath()
    PredictDataPath  = workDirsManager.getNameExistPath(args.basedir, args.predictionsdir)
    OriginMasksPath  = workDirsManager.getNameExistPath(BaseDataPath, nameOriginMasksRelPath)

    listPredictMasksFiles = findFilesDir(PredictDataPath, namePredictionsFiles)
    listOriginMasksFiles  = findFilesDir(OriginMasksPath, nameOriginMasksFiles)

    nbPredictionsFiles = len(listPredictMasksFiles)

    if (nbPredictionsFiles == 0):
        message = "num Predictions found in dir \'%s\'" %(PredictDataPath)
        CatchErrorException(message)


    listFuns_Metrics = {imetrics: DICTAVAILMETRICFUNS(imetrics, use_in_Keras=False) for imetrics in args.listPostprocessMetrics}
    out_predictAccuracyFilename = joinpathnames(PredictDataPath, nameAccuracyPredictFiles)
    fout = open(out_predictAccuracyFilename, 'w')

    strheader = '/case/ ' + ' '.join(
        ['/%s/' % (key) for (key, _) in listFuns_Metrics.iteritems()]) + '\n'
    fout.write(strheader)



    for i, predict_masks_file in enumerate(listPredictMasksFiles):

        print('\'%s\'...' %(predict_masks_file))

        name_prefix_case = getExtractSubstringPattern(predict_masks_file, tempSearchInputFiles)

        for iterfile in listOriginMasksFiles:
            if name_prefix_case in iterfile:
                origin_masks_file = iterfile
        #endfor

        print("assigned to '%s'..." %(basename(origin_masks_file)))


        predict_masks_array = FileReader.getImageArray(predict_masks_file)
        origin_masks_array  = FileReader.getImageArray(origin_masks_file)

        print("Predictions masks array of size: %s..." % (str(predict_masks_array.shape)))


        if (args.thresholdOutProbMaps):
            print("Threshold probability maps to value %s..." % (args.thresholdValue))

            predict_masks_array = ThresholdImages.compute(predict_masks_array, args.thresholdValue)


        list_predictAccuracy = OrderedDict()

        for (key, value) in listFuns_Metrics.iteritems():
            accuracy_value = value(origin_masks_array, predict_masks_array)
            list_predictAccuracy[key] = accuracy_value
        # endfor


        # print list accuracies on screen
        for (key, value) in list_predictAccuracy.iteritems():
            print("Computed '%s': %s..." %(key, value))
        #endfor

        # print list accuracies in file
        strdata  = '\'%s\' ' %(name_prefix_case)
        strdata += ' '.join([str(value) for (_,value) in list_predictAccuracy.iteritems()])
        strdata += '\n'
        fout.write(strdata)


        # Save thresholded final prediction masks
        if (args.thresholdOutProbMaps and args.saveThresholdImages):
            print("Saving thresholded prediction masks, with dims: %s..." %(tuple2str(predict_masks_array.shape)))

            out_predictMasksFilename = joinpathnames(PredictDataPath, tempOutPredictMasksFilename(predict_masks_file))

            FileReader.writeImageArray(out_predictMasksFilename, predict_masks_array)
    #endfor

    #close list accuracies file
    fout.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--predictionsdir', default='Predictions')
    parser.add_argument('--listPostprocessMetrics', type=parseListarg, default=LISTPOSTPROCESSMETRICS)
    parser.add_argument('--thresholdOutProbMaps', type=str2bool, default=THRESHOLDOUTPROBMAPS)
    parser.add_argument('--thresholdValue', type=float, default=THRESHOLDVALUE)
    parser.add_argument('--saveThresholdImages', type=str2bool, default=SAVETHRESHOLDIMAGES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)