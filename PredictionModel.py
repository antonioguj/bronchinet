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
    ProcessDataPath = workDirsManager.getNameExistPath(BaseDataPath, 'ProcInputData')
    ModelsPath      = workDirsManager.getNameExistPath(args.basedir, 'Models')
    RawMasksPath    = workDirsManager.getNameExistPath(BaseDataPath, 'ProcMasks')
    PredictDataPath = workDirsManager.getNameNewPath(args.basedir, 'Predictions')
    PredictDataPath = workDirsManager.getNameNewPath(PredictDataPath, workDirsManager.getNameRelDataPath(args.typedata))

    # Get the file list:
    nameImagesFiles   = 'images*'+ getFileExtension(FORMATINOUTDATA)
    nameMasksFiles    = 'masks*' + getFileExtension(FORMATINOUTDATA)
    nameOrigMasksFiles= '*.nii'

    listImagesFiles   = findFilesDir(ProcessDataPath, nameImagesFiles)
    listMasksFiles    = findFilesDir(ProcessDataPath, nameMasksFiles )
    listOrigMasksFiles= findFilesDir(RawMasksPath,    nameOrigMasksFiles)

    tempNamePredictionsFiles = 'predictMasks-%0.2i_acc%0.2f_dim'


    print("-" * 30)
    print("Loading saved model...")
    print("-" * 30)

    # Loading Saved Model
    modelSavedPath = joinpathnames(ModelsPath, getSavedModelFileName(args.prediction_modelFile))
    custom_objects = {'compute_loss': DICTAVAILLOSSFUNS(ILOSSFUN).compute_loss,
                      'compute': DICTAVAILMETRICS(IMETRICS).compute}

    model = NeuralNetwork.getLoadSavedModel(modelSavedPath, custom_objects=custom_objects)

    PredictAccuracyMetrics = DICTAVAILMETRICS(args.predictAccuracyMetrics)


    print("-" * 30)
    print("Predicting model...")
    print("-" * 30)

    for i, (imagesFile, masksFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

        print('\'%s\'...' % (imagesFile))

        # Loading Data
        if (args.multiClassCase):
            loadDataManager = LoadDataManager(IMAGES_DIMS_Z_X_Y, num_classes_out=args.numClassesMasks+1)
        else:
            loadDataManager = LoadDataManager(IMAGES_DIMS_Z_X_Y)

        (xTest, yTest) = loadDataManager.loadData_1File_BatchGenerator(SlidingWindowImages(IMAGES_DIMS_Z_X_Y, args.prop_overlap_Z_X_Y),
                                                                       imagesFile, masksFile, shuffle_images=False)


        # Evaluate Model
        yPredict = model.predict(xTest, batch_size=1)

        # Compute test accuracy
        accuracy = PredictAccuracyMetrics.compute_np_safememory(yTest, yPredict)

        print("Computed accuracy: %s..."%(accuracy))


        # Reconstruct batch images to full 3D array
        predictions_array_shape = FileReader.getImageSize(listOrigMasksFiles[i])

        if (args.slidingWindowImages):
            reconstructorImage = ReconstructorImage(predictions_array_shape, IMAGES_DIMS_Z_X_Y, prop_overlap=args.prop_overlap_Z_X_Y)
        else:
            reconstructorImage = ReconstructorImage(predictions_array_shape, IMAGES_DIMS_Z_X_Y)

        predictions_array = reconstructorImage.compute(yPredict.reshape([yPredict.shape[0], IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH]))


        # Save predictions data
        print("Saving predictions data, with dims: %s..." %(tuple2str(predictions_array.shape)))

        out_predictionsFilename = joinpathnames(PredictDataPath, tempNamePredictionsFiles%(i, accuracy) + tuple2str(predictions_array.shape) + getFileExtension(FORMATINOUTDATA))

        FileReader.writeImageArray(out_predictionsFilename, predictions_array)


        if (args.saveVisualPredictData):
            print("Saving predictions data in image format for visualization...")

            out_predictionsFilename = joinpathnames(PredictDataPath, tempNamePredictionsFiles%(i, accuracy) + tuple2str(predictions_array.shape) + '.nii')

            FileReader.writeImageArray(out_predictionsFilename, predictions_array)
    #endfor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--typedata', default=TYPEDATA)
    parser.add_argument('--multiClassCase', type=str2bool, default=MULTICLASSCASE)
    parser.add_argument('--numClassesMasks', type=int, default=NUMCLASSESMASKS)
    parser.add_argument('--prediction_modelFile', default=PREDICTION_MODELFILE)
    parser.add_argument('--prediction_epoch', default='last')
    parser.add_argument('--predictAccuracyMetrics', default=PREDICTACCURACYMETRICS)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    parser.add_argument('--saveVisualPredictData', type=str2bool, default=SAVEVISUALPREDICTDATA)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
