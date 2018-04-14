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
from CommonUtil.FileReaders import *
from CommonUtil.FunctionsUtil import *
from CommonUtil.LoadDataManager import *
from CommonUtil.PlotsManager import *
from CommonUtil.WorkDirsManager import *
from Networks.Metrics import *
from Networks.Networks import *
from Preprocessing.OperationsImages import *
from Preprocessing.SlidingWindowImages import *
from Postprocessing.ReconstructorImages import *
import argparse


def main(args):

    workDirsManager = WorkDirsManager(args.basedir)
    BaseDataPath    = workDirsManager.getNameDataPath(args.typedata)
    ProcessDataPath = workDirsManager.getNameNewPath(BaseDataPath, 'ProcessInputData')
    ModelsPath      = workDirsManager.getNameNewPath(args.basedir, 'Models')
    PredictDataPath = workDirsManager.getNameNewPath(BaseDataPath, 'ProcessOutputData')

    # Get the file list:
    nameImagesFiles = 'images*'+ getFileExtension(FORMATINOUTDATA)
    nameMasksFiles  = 'masks*' + getFileExtension(FORMATINOUTDATA)

    listProcImagesFiles = findFilesDir(ProcessDataPath, nameImagesFiles)
    listProcMasksFiles  = findFilesDir(ProcessDataPath, nameMasksFiles )

    tempNamePredictionsFiles = 'predictions-%0.2i_acc%0.2f_dim'

    if (args.saveVisualPredictData):
        VisualProcDataPath = workDirsManager.getNameNewPath(BaseDataPath, 'VisualProcessData')


    print("-" * 30)
    print("Loading saved model...")
    print("-" * 30)

    # Loading Saved Model
    modelSavedPath = joinpathnames(ModelsPath, getSavedModelFileName(args.prediction_modelFile))
    custom_objects = {'compute_loss': DICTAVAILLOSSFUNS[ILOSSFUN].compute_loss,
                      'compute': DICTAVAILMETRICS[IMETRICS].compute}

    model = NeuralNetwork.getLoadSavedModel(modelSavedPath, custom_objects=custom_objects)

    PredictAccuracyMetrics = DICTAVAILMETRICS[args.predictionAccuracy]


    print("-" * 30)
    print("Predicting model...")
    print("-" * 30)

    for i, (imagesFile, masksFile) in enumerate(zip(listProcImagesFiles, listProcMasksFiles)):

        print('\'%s\'...' % (imagesFile))

        # Loading Data
        (xTest, yTest) = LoadDataManager(IMAGES_DIMS_Z_X_Y).loadData_1File_BatchGenerator(SlidingWindowImages(IMAGES_DIMS_Z_X_Y, args.prop_overlap_Z_X_Y),
                                                                                          imagesFile, masksFile, shuffle_images=False)

        # Evaluate Model
        yPredict = model.predict(xTest, batch_size=1)

        # Compute test accuracy
        accuracy = PredictAccuracyMetrics.compute_home(yPredict, yTest)

        print("Computed accuracy: %s..." %(accuracy))


        if (args.confineMasksToLungs):
            print("Confine masks to exclude the area outside the lungs...")

            yPredict = ExclusionMasks.computeInverse(yPredict, yTest)


        # Reconstruct batch images to full 3D array
        predictions_array_shape = FileReader.getImageSize(listProcImagesFiles[i])

        if (args.slidingWindowImages):
            reconstructorImage = ReconstructorImage(predictions_array_shape, IMAGES_DIMS_Z_X_Y, prop_overlap=args.prop_overlap_Z_X_Y)
        else:
            reconstructorImage = ReconstructorImage(predictions_array_shape, IMAGES_DIMS_Z_X_Y)

        predictions_array = reconstructorImage.compute(yPredict.reshape([yPredict.shape[0], IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH]))


        # Save predictions data
        print("Saving processed data, with dims: %s..." %(tuple2str(predictions_array.shape)))

        out_predictionsFilename = joinpathnames(PredictDataPath, tempNamePredictionsFiles%(i, accuracy) + tuple2str(predictions_array.shape) + getFileExtension(FORMATINOUTDATA))

        FileReader.writeImageArray(out_predictionsFilename, predictions_array)


        if (args.saveVisualPredictData):
            print("Saving processed data in image format for visualization...")

            out_predictionsFilename = joinpathnames(VisualProcDataPath, tempNamePredictionsFiles%(i, accuracy) + tuple2str(predictions_array.shape) + '.nii')

            FileReader.writeImageArray(out_predictionsFilename, predictions_array)
    #endfor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--typedata', default=TYPEDATA)
    parser.add_argument('--prediction_modelFile', default=PREDICTION_MODELFILE)
    parser.add_argument('--predictionAccuracy', default=PREDICTIONACCURACY)
    parser.add_argument('--confineMasksToLungs', default=CONFINEMASKSTOLUNGS)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    parser.add_argument('--saveVisualPredictData', type=str2bool, default=SAVEVISUALPREDICTDATA)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)