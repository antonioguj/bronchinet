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

    workDirsManager  = WorkDirsManager(args.basedir)
    BaseDataPath     = workDirsManager.getNameBaseDataPath()
    PredictDataPath  = workDirsManager.getNameExistPath(args.basedir, args.predictionsdir)
    OriginMasksPath  = workDirsManager.getNameExistPath(BaseDataPath, 'ProcMasks')

    # Get the file list:
    namePredictionsFiles  = 'predictMasks*.nii.gz'
    nameOriginMasksFiles  = '*.nii.gz'

    listPredictMasksFiles = findFilesDir(PredictDataPath,  namePredictionsFiles)
    listOriginMasksFiles  = findFilesDir(OriginMasksPath,  nameOriginMasksFiles)

    nbPredictionsFiles = len(listPredictMasksFiles)

    if (nbPredictionsFiles == 0):
        message = "num Predictions found in dir \'%s\'" %(PredictDataPath)
        CatchErrorException(message)


    listFuns_Metrics = {imetrics: DICTAVAILMETRICFUNS(imetrics, use_in_Keras=False) for imetrics in args.listPostprocessMetrics}

    # create file to save accuracy measures on test sets
    if (args.thresholdOutProbMaps):
        nameAccuracyPredictFiles = 'predictAccuracyTests_thres%s.txt'%(args.thresholdValue)
    else:
        nameAccuracyPredictFiles = 'predictAccuracyTests.txt'

    out_predictAccuracyFilename = joinpathnames(PredictDataPath, nameAccuracyPredictFiles)
    fout = open(out_predictAccuracyFilename, 'w')

    strheader = '/case/ ' + ' '.join(
        ['/%s/' % (key) for (key, _) in listFuns_Metrics.iteritems()]) + '\n'
    fout.write(strheader)



    for i, predict_masksFile in enumerate(listPredictMasksFiles):

        print('\'%s\'...' %(predict_masksFile))

        name_prefix_case = getExtractSubstringPattern(predict_masksFile, 'av[0-9]*')

        for iterfile in listOriginMasksFiles:
            if name_prefix_case in iterfile:
                origin_masksFile = iterfile
        #endfor

        print("assigned to '%s'..." %(basename(origin_masksFile)))


        predict_masks_array = FileReader.getImageArray(predict_masksFile)
        origin_masks_array  = FileReader.getImageArray(origin_masksFile)

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

            out_predictMasksFilename = joinpathnames(PredictDataPath, filenamenoextension(predict_masksFile) + 'thres%s.nii.gz' %(args.thresholdValue))

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
    parser.add_argument('--saveThresholdImages', default=SAVETHRESHOLDIMAGES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)