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
from Prototypes.SlidingWindowBatchGenerator import *
from keras import callbacks as Kcallbacks
import time



#MAIN
workDirsManager    = WorkDirsManager(BASEDIR)
TrainingDataPath   = workDirsManager.getNameNewPath(workDirsManager.getNameTrainingDataPath(), 'ProcVolsData')
ValidationDataPath = workDirsManager.getNameNewPath(workDirsManager.getNameValidationDataPath(), 'ProcVolsData')
ModelsPath         = workDirsManager.getNameModelsPath()


# Get the file list:
listTrainImagesFiles = findFilesDir(TrainingDataPath + '/volsImages*.npy')
listTrainMasksFiles  = findFilesDir(TrainingDataPath + '/volsMasks*.npy' )
listValidImagesFiles = findFilesDir(ValidationDataPath + '/volsImages*.npy')
listValidMasksFiles  = findFilesDir(ValidationDataPath + '/volsMasks*.npy' )


# LOADING DATA
# ----------------------------------------------
print('-' * 30)
print('Loading data...')
print('-' * 30)

(xTrain, yTrain) = LoadDataManager(IMAGES_DIMS_Z_X_Y).loadData_ListFiles(listTrainImagesFiles, listTrainMasksFiles)
(xValid, yValid) = LoadDataManager(IMAGES_DIMS_Z_X_Y).loadData_ListFiles_BatchGenerator(SlicingImages(IMAGES_DIMS_Z_X_Y), listValidImagesFiles, listValidMasksFiles)

print('Number Training volumes: %s' %(len(xTrain)))
print('Number Validation volumes: %s' %(xValid.shape[0]))


# BUILDING MODEL
# ----------------------------------------------
print('_' * 30)
print('Building model...')
print('_' * 30)

model = DICTAVAILNETWORKS3D[IMODEL].getModel(IMAGES_DIMS_Z_X_Y)

# Compile model
model.compile(optimizer= DICTAVAILOPTIMIZERS_USERLR(IOPTIMIZER, LEARN_RATE),
              loss     = WeightedBinaryCrossEntropy.compute_loss,
              metrics  =[DiceCoefficient.compute])
model.summary()

# Callbacks:
callbacks_list = []
callbacks_list.append(RecordLossHistory(ModelsPath))

filename = ModelsPath + '/weights.{epoch:02d}-{loss:.5f}.hdf5'
callbacks_list.append(callbacks.ModelCheckpoint(filename, monitor='loss', verbose=0, save_best_only=True))
filename = ModelsPath + '/best_weights.{epoch:02d}-{val_loss:.5f}.hdf5'
callbacks_list.append(callbacks.ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True))

#callbacks_list.append(Kcallbacks.EarlyStopping(monitor='val_loss', patience=10, mode='max'))
callbacks_list.append(Kcallbacks.TerminateOnNaN())
# ----------------------------------------------


# TRAINING MODEL
# ----------------------------------------------
print('-' * 30)
print('Training model...')
print('-' * 30)

if USE_RESTARTMODEL:
    print('Loading Weights and Restarting...')
    weightsPath = joinpathnames(ModelsPath, 'bestModel.hdf5')
    model.load_weights(weightsPath)

starttime = time.time()

if (USE_DATAAUGMENTATION):
    if (SLIDINGWINDOWIMAGES):
        # Images Data Generator by Sliding-window
        batchDataGenerator = SlidingWindowBatchGenerator(xTrain, yTrain, IMAGES_DIMS_Z_X_Y, PROP_OVERLAP_Z_X_Y, batch_size=1, shuffle=True)
        model_info = model.fit_generator(batchDataGenerator,
                                         steps_per_epoch=len(batchDataGenerator),
                                         nb_epoch=NBEPOCHS,
                                         verbose=1,
                                         shuffle=True,
                                         validation_data=(xValid, yValid),
                                         callbacks=callbacks_list)
    else:
        message = "Data augmentation model non existing..."
        CatchErrorException(message)
else:
    model_info = model.fit(xTrain, yTrain,
                           batch_size=1, # BATCH_SIZE
                           epochs=NBEPOCHS,
                           verbose=1,
                           shuffle=True,
                           validation_data=(xValid, yValid),
                           callbacks=callbacks_list)

endtime = time.time()

print('Training finished in %f' %(endtime - starttime))
# ----------------------------------------------
