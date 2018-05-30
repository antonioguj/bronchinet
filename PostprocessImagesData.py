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
import argparse


def main(args):

    workDirsManager = WorkDirsManager(args.basedir)
    BaseDataPath    = workDirsManager.getNameBaseDataPath()
    PredictDataPath = workDirsManager.getNameExistPath(args.basedir, args.predictionsdir)
    RawImagesPath   = workDirsManager.getNameExistPath(BaseDataPath, 'ProcImages')
    RawMasksPath    = workDirsManager.getNameExistPath(BaseDataPath, 'ProcMasks')

    # Get the file list:
    namePredictionsFiles= 'predict-masks*' + getFileExtension(FORMATINOUTDATA)
    nameOrigImagesFiles = '*.nii'
    nameOrigMasksFiles  = '*.nii'

    listPredictionsFiles= findFilesDir(PredictDataPath, namePredictionsFiles)
    listOrigImagesFiles = findFilesDir(RawImagesPath, nameOrigImagesFiles)
    listOrigMasksFiles  = findFilesDir(RawMasksPath, nameOrigMasksFiles)

    nbPredictionsFiles = len(listPredictionsFiles)

    tempNamePredictMasksFiles = 'predictMasks-%s_acc%0.2f.nii'

    # Run checkers
    if (nbPredictionsFiles == 0):
        message = "num Predictions found in dir \'%s\'" %(listPredictionsFiles)
        CatchErrorException(message)

    if (args.confineMasksToLungs):

        AddMasksPath = workDirsManager.getNameExistPath(BaseDataPath, 'ProcAddMasks')

        nameAddMasksFiles = '*.nii'
        listAddMasksFiles = findFilesDir(AddMasksPath, nameAddMasksFiles)


    listFuns_computeAccuracy = {imetrics:DICTAVAILMETRICS(imetrics, use_in_Keras=False) for imetrics in POSTPROCESSIMAGEMETRICS}


    for i, predictionsFile in enumerate(listPredictionsFiles):

        print('\'%s\'...' %(predictionsFile))

        predictions_array = FileReader.getImageArray(predictionsFile)


        index_orig_images = getIndexOrigImagesFile(basename(predictionsFile), beginString='predict-masks')

        print('\'%s\'...' %(listOrigMasksFiles[index_orig_images]))

        masks_array = FileReader.getImageArray(listOrigMasksFiles[index_orig_images])

        if (predictions_array.shape != masks_array.shape):
            message = "size of predictions: %s, not equal to size of masks: %s..." %(predictions_array.shape, masks_array.shape)
            CatchErrorException(message)
        print("Predictions masks of size: %s..." % (str(predictions_array.shape)))


        if (args.multiClassCase):
            # Check the correct multilabels in "masks_array"
            if not checkCorrectNumClassesInMasks(masks_array, args.numClassesMasks):
                message = "In multiclass case, found wrong values in masks array: %s..." %(np.unique(masks_array))
                CatchErrorException(message)
        else:
            # Turn to binary masks (0, 1)
            masks_array = processBinaryMasks(masks_array)


        if (args.confineMasksToLungs):
            print("Confine masks to exclude the area outside the lungs...")

            exclude_masks_array = FileReader.getImageArray(listAddMasksFiles[index_orig_images])

            predictions_array= ExclusionMasks.computeInverse(predictions_array, exclude_masks_array)
            masks_array      = ExclusionMasks.computeInverse(masks_array,       exclude_masks_array)


        # Compute accuracy measures
        for (value, key) in listFuns_computeAccuracy.iteritems():
            accuracy = key(masks_array, predictions_array)
            print("Computed accuracy: %s : %s..." %(value, accuracy))
            if "DiceCoefficient" in value:
                accudice_out = accuracy
        #endfor


        # Save reconstructed prediction masks
        print("Saving prediction masks, with dims: %s..." %(tuple2str(predictions_array.shape)))

        out_predictMasksFilename = joinpathnames(PredictDataPath, tempNamePredictMasksFiles%(filenamenoextension(listOrigImagesFiles[index_orig_images]), accudice_out))

        FileReader.writeImageArray(out_predictMasksFilename, predictions_array)


        # Save predictions in images
        if (args.savePredictionImages):
            SaveImagesPath = workDirsManager.getNameNewPath(PredictDataPath, 'imagesSlices-%s'%(filenamenoextension(listOrigImagesFiles[index_orig_images])))

            images_array = FileReader.getImageArray(listOrigImagesFiles[index_orig_images])

            #take only slices in the middle of lungs (1/5 - 4/5)*depth_Z
            begin_slices = images_array.shape[0] // 5
            end_slices   = 4 * images_array.shape[0] // 5

            PlotsManager.plot_compare_images_masks_allSlices(images_array[begin_slices:end_slices],
                                                             masks_array [begin_slices:end_slices],
                                                             predictions_array[begin_slices:end_slices],
                                                             isSaveImages=True, outfilespath=SaveImagesPath)
    #endfor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--predictionsdir', default='Predictions')
    parser.add_argument('--multiClassCase', type=str2bool, default=MULTICLASSCASE)
    parser.add_argument('--numClassesMasks', type=int, default=NUMCLASSESMASKS)
    parser.add_argument('--confineMasksToLungs', default=CONFINEMASKSTOLUNGS)
    parser.add_argument('--prediction_epoch', default='last')
    parser.add_argument('--savePredictionImages', default=SAVEPREDICTIONIMAGES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)