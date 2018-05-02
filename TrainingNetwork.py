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
from CommonUtil.KerasBatchDataGenerator import *
from CommonUtil.LoadDataManager import *
from CommonUtil.WorkDirsManager import *
from Networks.Callbacks import *
from Networks.Metrics import *
from Networks.Networks import *
from Networks.Optimizers import *
from Preprocessing.BaseImageGenerator import *
from keras import callbacks as Kcallbacks
import argparse


def main(args):

    workDirsManager    = WorkDirsManager(args.basedir)
    TrainingDataPath   = workDirsManager.getNameExistPath(workDirsManager.getNameTrainingDataPath())
    ValidationDataPath = workDirsManager.getNameExistPath(workDirsManager.getNameValidationDataPath())
    ModelsPath         = workDirsManager.getNameNewPath(args.basedir, 'Models')

    # Get the file list:
    nameImagesFiles = 'images*'+ getFileExtension(FORMATINOUTDATA)
    nameMasksFiles  = 'masks*' + getFileExtension(FORMATINOUTDATA)

    listTrainImagesFiles = findFilesDir(TrainingDataPath,   nameImagesFiles)
    listTrainMasksFiles  = findFilesDir(TrainingDataPath,   nameMasksFiles )
    listValidImagesFiles = findFilesDir(ValidationDataPath, nameImagesFiles)
    listValidMasksFiles  = findFilesDir(ValidationDataPath, nameMasksFiles )

    if not listValidImagesFiles or not listValidMasksFiles:
        use_validation_data = False
        message = "No Validation Data used for training network..."
        CatchWarningException(message)
    else:
        use_validation_data = True



    # BUILDING MODEL
    # ----------------------------------------------
    print("_" * 30)
    print("Building model...")
    print("_" * 30)

    if args.use_restartModel:
        print('Loading saved model and Restarting...')
        initial_epoch = args.epoch_restart

        modelSavedPath = joinpathnames(ModelsPath, getSavedModelFileName(args.restart_modelFile))
        custom_objects = {'compute_loss': DICTAVAILLOSSFUNS(args.lossfun),
                          'compute': DICTAVAILMETRICS(args.metrics)}

        model = NeuralNetwork.getLoadSavedModel(modelSavedPath, custom_objects=custom_objects)
    else:
        initial_epoch = 0

        if (args.multiClassCase):
            modelConstructor = DICTAVAILNETWORKS3D(IMAGES_DIMS_Z_X_Y, args.model, args.numClassesMasks+1)
        else:
            modelConstructor = DICTAVAILNETWORKS3D(IMAGES_DIMS_Z_X_Y, args.model)

        model = modelConstructor.getModel()

        # Compile model
        model.compile(optimizer= DICTAVAILOPTIMIZERS_USERLR(args.optimizer, args.learn_rate),
                      loss     = DICTAVAILLOSSFUNS(args.lossfun),
                      metrics  = [DICTAVAILMETRICS(args.metrics)])
    model.summary()

    # Callbacks:
    callbacks_list = []
    callbacks_list.append(RecordLossHistory(ModelsPath))

    filename = ModelsPath + '/model_{epoch:02d}_{loss:.5f}_{val_loss:.5f}.hdf5'
    callbacks_list.append(callbacks.ModelCheckpoint(filename, monitor='loss', verbose=0))
    #callbacks_list.append(Kcallbacks.EarlyStopping(monitor='val_loss', patience=10, mode='max'))
    callbacks_list.append(Kcallbacks.TerminateOnNaN())
    # ----------------------------------------------



    # LOADING DATA
    # ----------------------------------------------
    print("-" * 30)
    print("Loading data...")
    print("-" * 30)

    if (args.multiClassCase):
        num_classes_out = args.numClassesMasks + 1
    else:
        num_classes_out = 1

    print("Load Training data...")
    if (args.slidingWindowImages or args.transformationImages):

        (train_xData, train_yData) = LoadDataManager.loadData_ListFiles(listTrainImagesFiles, listTrainMasksFiles)

        train_images_generator = BaseImageGenerator.getImagesGenerator(args.slidingWindowImages, args.prop_overlap_Z_X_Y, args.transformationImages)

        train_batch_data_generator = KerasTrainingBatchDataGenerator(IMAGES_DIMS_Z_X_Y,
                                                                     train_xData,
                                                                     train_yData,
                                                                     train_images_generator,
                                                                     num_classes_out=num_classes_out,
                                                                     batch_size=args.batch_size,
                                                                     shuffle=True)

        print("Number volumes: %s. Total Data batches generated: %s..." %(len(listTrainImagesFiles), len(train_batch_data_generator)))
    else:
        (train_xData, train_yData) = LoadDataManagerInBatches(IMAGES_DIMS_Z_X_Y).loadData_ListFiles(listTrainImagesFiles, listTrainMasksFiles)

        print("Number volumes: %s. Total Data batches generated: %s..." %(len(listTrainImagesFiles), len(train_xData)))


    if use_validation_data:
        print("Load Validation data...")
        if (args.slidingWindowImages or args.transformationImages):

            (valid_xData, valid_yData) = LoadDataManager.loadData_ListFiles(listValidImagesFiles, listValidMasksFiles)

            valid_images_generator = BaseImageGenerator.getImagesGenerator(args.slidingWindowImages, args.prop_overlap_Z_X_Y, args.transformationImages)

            valid_batch_data_generator = KerasTrainingBatchDataGenerator(IMAGES_DIMS_Z_X_Y,
                                                                         valid_xData,
                                                                         valid_yData,
                                                                         valid_images_generator,
                                                                         num_classes_out=num_classes_out,
                                                                         batch_size=args.batch_size,
                                                                         shuffle=True)
            validation_data = valid_batch_data_generator

            print("Number volumes: %s. Total Data batches generated: %s..." %(len(listValidImagesFiles), len(valid_batch_data_generator)))
        else:
            (valid_xData, valid_yData) = LoadDataManagerInBatches(IMAGES_DIMS_Z_X_Y).loadData_ListFiles(listValidImagesFiles, listValidMasksFiles)
            validation_data = (valid_xData, valid_yData)

            print("Number volumes: %s. Total Data batches generated: %s..." % (len(listTrainImagesFiles), len(valid_xData)))
    else:
        validation_data = None



    # TRAINING MODEL
    # ----------------------------------------------
    print("-" * 30)
    print("Training model...")
    print("-" * 30)

    if (args.slidingWindowImages or args.transformationImages):

        model.fit_generator(train_batch_data_generator,
                            steps_per_epoch=len(train_batch_data_generator),
                            nb_epoch=args.num_epochs,
                            verbose=1,
                            shuffle=True,
                            validation_data=validation_data,
                            callbacks=callbacks_list,
                            initial_epoch=initial_epoch)
    else:
        model.fit(train_xData, train_yData,
                  batch_size=args.batch_size,
                  epochs=args.num_epochs,
                  verbose=1,
                  shuffle=True,
                  validation_data=validation_data,
                  callbacks=callbacks_list,
                  initial_epoch=initial_epoch)
    # ----------------------------------------------


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--multiClassCase', type=str2bool, default=MULTICLASSCASE)
    parser.add_argument('--numClassesMasks', type=int, default=NUMCLASSESMASKS)
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--model', default=IMODEL)
    parser.add_argument('--optimizer', default=IOPTIMIZER)
    parser.add_argument('--lossfun', default=ILOSSFUN)
    parser.add_argument('--metrics', default=IMETRICS)
    parser.add_argument('--learn_rate', type=float, default=LEARN_RATE)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    parser.add_argument('--transformationImages', type=str2bool, default=TRANSFORMATIONIMAGES)
    parser.add_argument('--use_restartModel', type=str2bool, default=USE_RESTARTMODEL)
    parser.add_argument('--restart_modelFile', default=RESTART_MODELFILE)
    parser.add_argument('--epoch_restart', type=int, default=EPOCH_RESTART)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)