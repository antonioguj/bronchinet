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
    OriginImagesPath = workDirsManager.getNameExistPath(BaseDataPath, 'RawImages')
    OriginMasksPath  = workDirsManager.getNameExistPath(BaseDataPath, 'RawMasks')

    # Get the file list:
    namePredictionsFiles  = 'predict-masks*' + getFileExtension(FORMATINOUTDATA)
    nameOriginImagesFiles = '*.dcm'
    nameOriginMasksFiles  = '*.dcm'

    listPredictionsFiles  = findFilesDir(PredictDataPath,  namePredictionsFiles)
    listOriginImagesFiles = findFilesDir(OriginImagesPath, nameOriginImagesFiles)
    listOriginMasksFiles  = findFilesDir(OriginMasksPath,  nameOriginMasksFiles)

    nbPredictionsFiles = len(listPredictionsFiles)

    tempNamePredictMasksFiles = 'predictMasks-%s_acc%0.2f.nii'


    # Run checkers
    if (nbPredictionsFiles == 0):
        message = "num Predictions found in dir \'%s\'" %(PredictDataPath)
        CatchErrorException(message)

    if (args.cropImages):
        namefile_dict = joinpathnames(BaseDataPath, "boundBoxesMasks.npy")
        dict_masks_boundingBoxes = readDictionary(namefile_dict)

    if (args.confineMasksToLungs):

        OriginAddMasksPath = workDirsManager.getNameExistPath(BaseDataPath, 'RawAddMasks')

        nameAddMasksFiles = '*.dcm'
        listAddMasksFiles = findFilesDir(OriginAddMasksPath, nameAddMasksFiles)


    computePredictAccuracy = DICTAVAILMETRICFUNS(args.predictAccuracyMetrics, use_in_Keras=False)

    listFuns_computePredictAccuracy = {imetrics:DICTAVAILMETRICFUNS(imetrics, use_in_Keras=False) for imetrics in args.listPostprocessMetrics}


    # create file to save accuracy measures on test sets
    nameAccuracyPredictFiles = 'predictAccuracyTests.txt'

    out_predictAccuracyFilename = joinpathnames(PredictDataPath, nameAccuracyPredictFiles)
    fout = open(out_predictAccuracyFilename, 'w')

    strheader = '/case/ '+ ' '.join(['/%s/' %(key) for (key,_) in listFuns_computePredictAccuracy.iteritems()]) + '\n'
    fout.write(strheader)



    for i, predictionsFile in enumerate(listPredictionsFiles):

        print('\'%s\'...' %(predictionsFile))

        index_orig_images = getIndexOriginImagesFile(basename(predictionsFile), beginString='predict-masks')

        imagesFile = listOriginImagesFiles[index_orig_images]
        masksFile  = listOriginMasksFiles [index_orig_images]

        print("assigned to '%s' and '%s'..." %(basename(imagesFile), basename(masksFile)))


        predictions_array = FileReader.getImageArray(predictionsFile)
        masks_array       = FileReader.getImageArray(masksFile)

        print("Predictions masks array of size: %s..." % (str(predictions_array.shape)))


        if (args.multiClassCase):
            # Check the correct multilabels in "masks_array"
            if not checkCorrectNumClassesInMasks(masks_array, args.numClassesMasks):
                message = "In multiclass case, found wrong values in masks array: %s..." %(np.unique(masks_array))
                CatchErrorException(message)
        else:
            # Turn to binary masks (0, 1)
            masks_array = processBinaryMasks(masks_array)


        if (args.cropImages):
            if (predictions_array.shape > masks_array.shape):
                message = "size of predictions array: %s, cannot be larger than that of original masks: %s..." % (predictions_array.shape, masks_array.shape)
                CatchErrorException(message)
            else:
                print("Predictions are cropped. Augment size of predictions array from %s to original size %s..." %(predictions_array.shape, masks_array.shape))

            crop_boundingBox = dict_masks_boundingBoxes[filenamenoextension(imagesFile)]

            new_predictions_array = np.zeros(masks_array.shape)
            new_predictions_array[crop_boundingBox[0][0]:crop_boundingBox[0][1],
                                  crop_boundingBox[1][0]:crop_boundingBox[1][1],
                                  crop_boundingBox[2][0]:crop_boundingBox[2][1]] = predictions_array

            predictions_array = new_predictions_array
        else:
            if (predictions_array.shape != masks_array.shape):
                message = "size of predictions array: %s, not equal to size of masks: %s..." %(predictions_array.shape, masks_array.shape)
                CatchErrorException(message)


        if (args.thresholdOutProbMaps):
            print("Threshold probability maps to value %s..." %(args.thresholdValue))

            predictions_array = ThresholdImages.compute(predictions_array, args.thresholdValue)


        if (args.confineMasksToLungs):
            print("Confine masks to exclude the area outside the lungs...")

            exclude_masksFile   = listAddMasksFiles[index_orig_images]
            exclude_masks_array = FileReader.getImageArray(exclude_masksFile)

            masks_array_with_exclusion = ExclusionMasks.compute(masks_array, exclude_masks_array)

            predictions_array = ExclusionMasks.computeInverse(predictions_array, exclude_masks_array)

            # Compute accuracy measures
            accuracy = computePredictAccuracy(masks_array_with_exclusion, predictions_array)

            list_predictAccuracy = OrderedDict()

            for (key, value) in listFuns_computePredictAccuracy.iteritems():
                accuracy_value = value(masks_array_with_exclusion, predictions_array)
                list_predictAccuracy[key] = accuracy_value
            #endfor
        else:
            # Compute accuracy measures
            accuracy = computePredictAccuracy(masks_array, predictions_array)

            list_predictAccuracy = OrderedDict()

            for (key, value) in listFuns_computePredictAccuracy.iteritems():
                accuracy_value = value(masks_array, predictions_array)
                list_predictAccuracy[key] = accuracy_value
            #endfor


        # print list accuracies on screen
        for (key, value) in list_predictAccuracy.iteritems():
            print("Computed '%s': %s..." %(key, value))
        #endfor

        # print list accuracies in file
        strdata  = '\'%s\' ' %(filenamenoextension(imagesFile))
        strdata += ' '.join([str(value) for (_,value) in list_predictAccuracy.iteritems()])
        strdata += '\n'
        fout.write(strdata)


        # Save reconstructed prediction masks
        print("Saving prediction masks, with dims: %s..." %(tuple2str(predictions_array.shape)))

        out_predictMasksFilename = joinpathnames(PredictDataPath, tempNamePredictMasksFiles%(filenamenoextension(imagesFile), accuracy))

        FileReader.writeImageArray(out_predictMasksFilename, predictions_array)


        # Save predictions in images
        if (args.savePredictionImages):
            SaveImagesPath = workDirsManager.getNameNewPath(PredictDataPath, 'imagesSlices-%s'%(filenamenoextension(imagesFile)))

            images_array = FileReader.getImageArray(imagesFile)

            #take only slices in the middle of lungs (1/5 - 4/5)*depth_Z
            begin_slices = images_array.shape[0] // 5
            end_slices   = 4 * images_array.shape[0] // 5

            PlotsManager.plot_compare_images_masks_allSlices(images_array[begin_slices:end_slices],
                                                             masks_array [begin_slices:end_slices],
                                                             predictions_array[begin_slices:end_slices],
                                                             isSaveImages=True, outfilespath=SaveImagesPath)
    #endfor

    #close list accuracies file
    fout.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--predictionsdir', default='Predictions')
    parser.add_argument('--predictAccuracyMetrics', default=PREDICTACCURACYMETRICS)
    parser.add_argument('--listPostprocessMetrics', type=parseListarg, default=LISTPOSTPROCESSMETRICS)
    parser.add_argument('--multiClassCase', type=str2bool, default=MULTICLASSCASE)
    parser.add_argument('--numClassesMasks', type=int, default=NUMCLASSESMASKS)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    parser.add_argument('--thresholdOutProbMaps', type=str2bool, default=THRESHOLDOUTPROBMAPS)
    parser.add_argument('--thresholdValue', type=float, default=THRESHOLDVALUE)
    parser.add_argument('--confineMasksToLungs', default=CONFINEMASKSTOLUNGS)
    parser.add_argument('--prediction_epoch', default='last')
    parser.add_argument('--savePredictionImages', default=SAVEPREDICTIONIMAGES)
    args = parser.parse_args()

    if (args.confineMasksToLungs):
        args.predictAccuracyMetrics = args.predictAccuracyMetrics + '_Masked'
        args.listPostprocessMetrics = [item + '_Masked' for item in args.listPostprocessMetrics]

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)