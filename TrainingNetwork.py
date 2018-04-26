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
from CommonUtil.LoadDataManager import *
from CommonUtil.WorkDirsManager import *
from Networks.Callbacks import *
from Networks.Metrics import *
from Networks.Networks import *
from Networks.Optimizers import *
from KerasPrototypes.SlidingWindowBatchGenerator import *
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
        custom_objects = {'compute_loss': DICTAVAILLOSSFUNS(args.lossfun).compute_loss,
                          'compute': DICTAVAILMETRICS(args.metrics).compute}

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

    if (args.slidingWindowImages):

        def getDataBatchGenerator(listImagesFiles, listMasksFiles):

            (xData, yData) = LoadDataManager.loadData_ListFiles(listImagesFiles, listMasksFiles)

            if (args.slidingWindowImages):
                # Images Data Generator by Sliding-window...
                if (args.imagesAugmentation):
                    # Data augmentation by random transformation to input images...
                    return SlidingWindowPlusDataAugmentationBatchGenerator(xData, yData,
                                                                           IMAGES_DIMS_Z_X_Y,
                                                                           args.prop_overlap_Z_X_Y,
                                                                           num_classes_out=num_classes_out,
                                                                           batch_size=args.batch_size,
                                                                           shuffle=True,
                                                                           rotation_range=ROTATION_RANGE,
                                                                           width_shift_range=HORIZONTAL_SHIFT,
                                                                           height_shift_range=VERTICAL_SHIFT,
                                                                           horizontal_flip=HORIZONTAL_FLIP,
                                                                           vertical_flip=VERTICAL_FLIP)
                else:
                    return SlidingWindowBatchGenerator(xData, yData,
                                                       IMAGES_DIMS_Z_X_Y,
                                                       args.prop_overlap_Z_X_Y,
                                                       num_classes_out=num_classes_out,
                                                       batch_size=args.batch_size,
                                                       shuffle=True)


    print("Load Training data...")
    if (args.slidingWindowImages):

        trainData_generator = getDataBatchGenerator(listTrainImagesFiles, listTrainMasksFiles)

        print("Number volumes: %s. Total Data batches generated: %s..." %(len(listTrainImagesFiles), len(trainData_generator)))
    else:
        (xTrain, yTrain) = LoadDataManagerInBatches(IMAGES_DIMS_Z_X_Y).loadData_ListFiles(listTrainImagesFiles, listTrainMasksFiles)

        print("Number volumes: %s. Total Data batches generated: %s..." %(len(listTrainImagesFiles), len(xTrain)))


    if use_validation_data:
        print("Load Validation data...")
        if (args.slidingWindowImages):

            validData_generator = getDataBatchGenerator(listValidImagesFiles, listValidMasksFiles)
            validation_data = validData_generator

            print("Number volumes: %s. Total Data batches generated: %s..." %(len(listValidImagesFiles), len(validData_generator)))
        else:
            (xValid, yValid) = LoadDataManagerInBatches(IMAGES_DIMS_Z_X_Y).loadData_ListFiles(listValidImagesFiles, listValidMasksFiles)
            validation_data = (xValid, yValid)

            print("Number volumes: %s. Total Data batches generated: %s..." % (len(listTrainImagesFiles), len(xValid)))
    else:
        validation_data = None



    # TRAINING MODEL
    # ----------------------------------------------
    print("-" * 30)
    print("Training model...")
    print("-" * 30)

    if (args.slidingWindowImages):

        model.fit_generator(trainData_generator,
                            steps_per_epoch=len(trainData_generator),
                            nb_epoch=args.num_epochs,
                            verbose=1,
                            shuffle=True,
                            validation_data=validation_data,
                            callbacks=callbacks_list,
                            initial_epoch=initial_epoch)
    else:
        model.fit(xTrain, yTrain,
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
    parser.add_argument('--imagesAugmentation', type=str2bool, default=IMAGESAUGMENTATION)
    parser.add_argument('--use_restartModel', type=str2bool, default=USE_RESTARTMODEL)
    parser.add_argument('--restart_modelFile', default=RESTART_MODELFILE)
    parser.add_argument('--epoch_restart', type=int, default=EPOCH_RESTART)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)