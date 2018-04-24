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

    workDirsManager  = WorkDirsManager(args.basedir)
    BaseDataPath     = workDirsManager.getNameDataPath(args.typedata)
    PredictDataPath  = workDirsManager.getNameExistPath(args.basedir, joinpathnames('Predictions', workDirsManager.getNameRelDataPath(args.typedata)))
    RawImagesPath    = workDirsManager.getNameExistPath(BaseDataPath, 'ProcImages')
    RawMasksPath     = workDirsManager.getNameExistPath(BaseDataPath, 'ProcMasks')
    PredictionsPath  = workDirsManager.getNameNewPath(args.basedir, 'Predictions')

    # Get the file list:
    namePredictionsFiles= 'predictions*'+ getFileExtension(FORMATINOUTDATA)
    nameOrigImagesFiles = '*.nii'
    nameOrigMasksFiles  = '*.nii'

    listPredictionsFiles = findFilesDir(PredictDataPath, namePredictionsFiles)
    listOrigImagesFiles  = findFilesDir(RawImagesPath, nameOrigImagesFiles)
    listOrigMasksFiles   = findFilesDir(RawMasksPath, nameOrigMasksFiles)

    nbPredictionsFiles  = len(listPredictionsFiles)
    nbOrigMasksFiles    = len(listOrigMasksFiles)

    tempNamePredictMasksFiles = 'predictions-%s_acc%0.2f.nii'

    # Run checkers
    if (nbPredictionsFiles == 0):
        message = "num Predictions found in dir \'%s\'" %(listPredictionsFiles)
        CatchErrorException(message)
    if (nbPredictionsFiles != nbOrigMasksFiles):
        message = "num Predictions %i not equal to num Masks %i" %(nbPredictionsFiles, nbOrigMasksFiles)
        CatchErrorException(message)


    if (args.confineMasksToLungs):

        AddMasksPath = workDirsManager.getNameExistPath(BaseDataPath, 'ProcAddMasks')

        nameAddMasksFiles = '*.nii'
        listAddMasksFiles = findFilesDir(AddMasksPath, nameAddMasksFiles)
        nbAddMasksFiles   = len(listAddMasksFiles)

        if (nbPredictionsFiles != nbAddMasksFiles):
            message = "num Predictions %i not equal to num Masks %i" %(nbPredictionsFiles, nbAddMasksFiles)
            CatchErrorException(message)



    for i, (predictionsFile, masksFile) in enumerate(zip(listPredictionsFiles, listOrigMasksFiles)):

        print('\'%s\'...' %(predictionsFile))

        predictions_array = FileReader.getImageArray(predictionsFile)
        masks_array       = FileReader.getImageArray(masksFile)

        if (predictions_array.shape != masks_array.shape):
            message = "size of predictions: %s, not equal to size of masks: %s..." %(predictions_array.shape, masks_array.shape)
            CatchErrorException(message)
        print("Predictions masks of size: %s..." % (str(predictions_array.shape)))


        if (not args.multiClassCase):
            # Turn to binary masks (0, 1)
            masks_array = processBinaryMasks(masks_array)


        if (args.confineMasksToLungs):
            print("Confine masks to exclude the area outside the lungs...")

            exclude_masks_array = FileReader.getImageArray(listAddMasksFiles[i])

            predictions_array = ExclusionMasks.computeInverse(predictions_array, exclude_masks_array)


        # Compute accuracy measures
        accuracy_dice = DiceCoefficient().compute_np_safememory(masks_array, predictions_array)
        accuracy_TP   = TruePositiveRate().compute_np_safememory(masks_array, predictions_array)
        accuracy_FP   = FalsePositiveRate().compute_np_safememory(masks_array, predictions_array)
        accuracy_TN   = TrueNegativeRate().compute_np_safememory(masks_array, predictions_array)
        accuracy_FN   = FalseNegativeRate().compute_np_safememory(masks_array, predictions_array)

        print("Computed accuracy: dice coefficient: %s..."%(accuracy_dice))
        print("Computed accuracy: true positive rate: %s..." % (accuracy_TP))
        print("Computed accuracy: false positive rate: %s..." % (accuracy_FP))
        print("Computed accuracy: true negative rate: %s..." % (accuracy_TN))
        print("Computed accuracy: false negative rate: %s..." % (accuracy_FN))


        # Save reconstructed prediction masks
        print("Saving prediction masks, with dims: %s..." %(tuple2str(predictions_array.shape)))

        out_predictMasksFilename = joinpathnames(PredictionsPath, tempNamePredictMasksFiles%(filenamenoextension(masksFile), accuracy_dice))

        FileReader.writeImageArray(out_predictMasksFilename, predictions_array)


        # Save predictions in images
        if (args.savePredictionImages):
            SaveImagesPath = workDirsManager.getNameNewPath(PredictionsPath, 'imagesSlices-%s'%(filenamenoextension(masksFile)))

            images_array = FileReader.getImageArray(listOrigImagesFiles[i])

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
    parser.add_argument('--typedata', default=TYPEDATA)
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