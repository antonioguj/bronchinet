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
from CommonUtil.LoadDataManager import *
from CommonUtil.FunctionsUtil import *
from CommonUtil.WorkDirsManager import *
from Networks.Callbacks import *
from Networks.Metrics import *
from Networks.Networks import *
from Networks.Optimizers import *
from Preprocessing.SlidingWindowImages import *
from KerasPrototypes.SlidingWindowBatchGenerator import *
from keras import callbacks as Kcallbacks
import argparse


def main(args):

    workDirsManager    = WorkDirsManager(args.basedir)
    TrainingDataPath   = workDirsManager.getNameExistPath(workDirsManager.getNameTrainingDataPath(), 'ProcInputData')
    ValidationDataPath = workDirsManager.getNameExistPath(workDirsManager.getNameValidationDataPath(), 'ProcInputData')
    ModelsPath         = workDirsManager.getNameNewPath(args.basedir, 'Models')

    # Get the file list:
    nameImagesFiles = 'images*'+ getFileExtension(FORMATINOUTDATA)
    nameMasksFiles  = 'masks*' + getFileExtension(FORMATINOUTDATA)

    listTrainImagesFiles = findFilesDir(TrainingDataPath,   nameImagesFiles)
    listTrainMasksFiles  = findFilesDir(TrainingDataPath,   nameMasksFiles )
    listValidImagesFiles = findFilesDir(ValidationDataPath, nameImagesFiles)
    listValidMasksFiles  = findFilesDir(ValidationDataPath, nameMasksFiles )


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

        modelConstructor = DICTAVAILNETWORKS3D(IMAGES_DIMS_Z_X_Y, args.model)
        model = modelConstructor.getModel()

        # Compile model
        model.compile(optimizer= DICTAVAILOPTIMIZERS_USERLR(args.optimizer, args.learn_rate),
                      loss     = DICTAVAILLOSSFUNS(args.lossfun).compute_loss,
                      metrics  = [DICTAVAILMETRICS(args.metrics).compute])
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
        loadDataManager = LoadDataManager(IMAGES_DIMS_Z_X_Y, num_classes_out=args.numClassesMasks+1)
    else:
        loadDataManager = LoadDataManager(IMAGES_DIMS_Z_X_Y)

    (xTrain, yTrain) = loadDataManager.loadData_ListFiles(listTrainImagesFiles, listTrainMasksFiles)
    (xValid, yValid) = loadDataManager.loadData_ListFiles_BatchGenerator(SlidingWindowImages(IMAGES_DIMS_Z_X_Y, args.prop_overlap_Z_X_Y),
                                                                         listValidImagesFiles, listValidMasksFiles)

    print("Number Training volumes: %s" %(len(xTrain)))
    print("Number Validation volumes: %s" %(xValid.shape[0]))


    # TRAINING MODEL
    # ----------------------------------------------
    print("-" * 30)
    print("Training model...")
    print("-" * 30)

    if (args.use_dataAugmentation):
        if (args.slidingWindowImages):

            if (args.multiClassCase):
                num_classes_out = args.numClassesMasks
            else:
                num_classes_out = 1
            # Images Data Generator by Sliding-window
            batchDataGenerator = SlidingWindowBatchGenerator(xTrain, yTrain,
                                                             IMAGES_DIMS_Z_X_Y,
                                                             args.prop_overlap_Z_X_Y,
                                                             num_classes_out=num_classes_out,
                                                             batch_size=args.batch_size,
                                                             shuffle=True)
            model.fit_generator(batchDataGenerator,
                                steps_per_epoch=len(batchDataGenerator),
                                nb_epoch=args.num_epochs,
                                verbose=1,
                                shuffle=True,
                                validation_data=(xValid, yValid),
                                callbacks=callbacks_list,
                                initial_epoch=initial_epoch)
        else:
            message = "Data augmentation model non existing..."
            CatchErrorException(message)
    else:
        model.fit(xTrain, yTrain,
                  batch_size=args.batch_size,
                  epochs=args.num_epochs,
                  verbose=1,
                  shuffle=True,
                  validation_data=(xValid, yValid),
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
    parser.add_argument('--use_dataAugmentation', type=str2bool, default=USE_DATAAUGMENTATION)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    parser.add_argument('--use_restartModel', type=str2bool, default=USE_RESTARTMODEL)
    parser.add_argument('--restart_modelFile', default=RESTART_MODELFILE)
    parser.add_argument('--epoch_restart', type=int, default=EPOCH_RESTART)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)