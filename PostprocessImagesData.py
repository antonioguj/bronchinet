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
    PredictDataPath = workDirsManager.getNameExistPath(args.basedir, 'Predictions')
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
        accuracy_bin_cross    = BinaryCrossEntropy().compute_np_safememory(masks_array, predictions_array)
        accuracy_wei_bin_cross= WeightedBinaryCrossEntropy().compute_np_safememory(masks_array, predictions_array)
        accuracy_dice         = DiceCoefficient().compute_np_safememory(masks_array, predictions_array)
        accuracy_tpr          = TruePositiveRate().compute_np_safememory(masks_array, predictions_array)
        accuracy_fpr          = FalsePositiveRate().compute_np_safememory(masks_array, predictions_array)
        accuracy_tnr          = TrueNegativeRate().compute_np_safememory(masks_array, predictions_array)
        accuracy_fnr          = FalseNegativeRate().compute_np_safememory(masks_array, predictions_array)

        print("Computed accuracy: binary cross-entropy: %s..." % (accuracy_bin_cross))
        print("Computed accuracy: binary weighted cross-entropy: %s..." % (accuracy_wei_bin_cross))
        print("Computed accuracy: dice coefficient: %s..."%(accuracy_dice))
        print("Computed accuracy: true positive rate: %s..." % (accuracy_tpr))
        print("Computed accuracy: false positive rate: %s..." % (accuracy_fpr))
        print("Computed accuracy: true negative rate: %s..." % (accuracy_tnr))
        print("Computed accuracy: false negative rate: %s..." % (accuracy_fnr))


        # Save reconstructed prediction masks
        print("Saving prediction masks, with dims: %s..." %(tuple2str(predictions_array.shape)))

        out_predictMasksFilename = joinpathnames(PredictDataPath, tempNamePredictMasksFiles%(filenamenoextension(listOrigImagesFiles[index_orig_images]), accuracy_dice))

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