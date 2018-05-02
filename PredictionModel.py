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
from Preprocessing.BaseImageGenerator import *
from Postprocessing.ReconstructorImages import *
import argparse


def main(args):

    workDirsManager = WorkDirsManager(args.basedir)
    TestingDataPath = workDirsManager.getNameExistPath(workDirsManager.getNameDataPath(args.typedata))
    ModelsPath      = workDirsManager.getNameExistPath(args.basedir, 'Models')
    PredictDataPath = workDirsManager.getNameNewPath(args.basedir, 'Predictions')

    # Get the file list:
    nameImagesFiles = 'images*'+ getFileExtension(FORMATINOUTDATA)
    nameMasksFiles  = 'masks*' + getFileExtension(FORMATINOUTDATA)

    listImagesFiles = findFilesDir(TestingDataPath, nameImagesFiles)
    listMasksFiles  = findFilesDir(TestingDataPath, nameMasksFiles )

    tempNamePredictionsFiles = 'predict-%s_acc%0.2f'


    print("-" * 30)
    print("Loading saved model...")
    print("-" * 30)

    # Loading Saved Model
    modelSavedPath = joinpathnames(ModelsPath, getSavedModelFileName(args.prediction_modelFile))
    custom_objects = {'compute_loss': DICTAVAILLOSSFUNS(ILOSSFUN),
                      'compute': DICTAVAILMETRICS(IMETRICS)}

    model = NeuralNetwork.getLoadSavedModel(modelSavedPath, custom_objects=custom_objects)

    computePredictAccuracy = DICTAVAILMETRICS_2(args.predictAccuracyMetrics)


    print("-" * 30)
    print("Predicting model...")
    print("-" * 30)

    if (args.multiClassCase):
        num_classes_out = args.numClassesMasks + 1
    else:
        num_classes_out = 1


    for i, (imagesFile, masksFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

        print('\'%s\'...' % (imagesFile))

        # Loading Data
        if (args.slidingWindowImages or args.transformationImages):
            test_images_generator = BaseImageGenerator.getImagesGenerator(args.slidingWindowImages, args.prop_overlap_Z_X_Y, args.transformationImages)

            (test_xData, test_yData) = LoadDataManagerInBatches_DataGenerator(IMAGES_DIMS_Z_X_Y,
                                                                              test_images_generator,
                                                                              num_classes_out=num_classes_out).loadData_1File(imagesFile, masksFile, shuffle_images=False)
        else:
            (test_xData, test_yData) = LoadDataManagerInBatches(IMAGES_DIMS_Z_X_Y).loadData_1File(imagesFile, masksFile, shuffle_images=False)


        # Evaluate Model
        predict_data = model.predict(test_xData, batch_size=1)

        # Compute test accuracy
        accuracy = computePredictAccuracy(test_yData, predict_data)

        print("Computed accuracy: %s..."%(accuracy))


        # Reconstruct batch images to full 3D array
        predictMasks_array_shape = FileReader.getImageSize(masksFile)

        if (args.slidingWindowImages):
            reconstructorImages = ReconstructorImages3D(predictMasks_array_shape,
                                                        IMAGES_DIMS_Z_X_Y,
                                                        prop_overlap=args.prop_overlap_Z_X_Y,
                                                        num_classes_out=num_classes_out)
        else:
            reconstructorImages = ReconstructorImages3D(predictMasks_array_shape,
                                                        IMAGES_DIMS_Z_X_Y,
                                                        num_classes_out=num_classes_out)

        predictMasks_array = reconstructorImages.compute(predict_data)


        # Save predictions data
        print("Saving predictions data, with dims: %s..." %(tuple2str(predictMasks_array.shape)))

        out_predictionsFilename = joinpathnames(PredictDataPath, tempNamePredictionsFiles%(filenamenoextension(masksFile), accuracy) + getFileExtension(FORMATINOUTDATA))

        FileReader.writeImageArray(out_predictionsFilename, predictMasks_array)


        if (args.saveVisualPredictData):
            print("Saving predictions data in image format for visualization...")

            out_predictionsFilename = joinpathnames(PredictDataPath, tempNamePredictionsFiles%(filenamenoextension(masksFile), accuracy) + '.nii')

            FileReader.writeImageArray(out_predictionsFilename, predictMasks_array)
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
    parser.add_argument('--transformationImages', type=str2bool, default=TRANSFORMATIONIMAGES)
    parser.add_argument('--saveVisualPredictData', type=str2bool, default=SAVEVISUALPREDICTDATA)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)