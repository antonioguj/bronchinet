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
from CommonUtil.FileDataManager import *
from CommonUtil.WorkDirsManager import *
from CommonUtil.PlotsManager import *
from CommonUtil.ErrorMessages import *
from Networks.Callbacks import *
from Networks.Metrics import *
from Networks.Networks import *
from Networks.Optimizers import *
from keras import callbacks as Kcallbacks
from keras.preprocessing import image as Kpreprocessing
import time

NBEPOCHS   = 500
BATCH_SIZE = 12
LEARN_RATE = 1.0e-05

IMODEL     = 'Unet2D'
IOPTIMIZER = 'Adam'

USE_DATAAUGMENTATION = False

USE_RESTARTMODEL = False


#MAIN
workDirsManager    = WorkDirsManager(BASEDIR)
TrainingDataPath   = workDirsManager.getNameNewPath(workDirsManager.getNameTrainingDataPath(), 'ProcSlicesData')
ValidationDataPath = workDirsManager.getNameNewPath(workDirsManager.getNameValidationDataPath(), 'ProcSlicesData')
ModelsPath         = workDirsManager.getNameModelsPath()


# LOADING DATA
# ----------------------------------------------
print('-' * 30)
print('Loading data...')
print('-' * 30)

listTrainImagesFiles = sorted(glob(TrainingDataPath + '/slicesImages*.npy'))
listTrainMasksFiles  = sorted(glob(TrainingDataPath + '/slicesMasks*.npy' ))
listValidImagesFiles = sorted(glob(ValidationDataPath + '/slicesImages*.npy'))
listValidMasksFiles  = sorted(glob(ValidationDataPath + '/slicesMasks*.npy' ))

(xTrain, yTrain) = FileDataManager.loadDataListFiles2D(listTrainImagesFiles, listTrainMasksFiles)
(xValid, yValid) = FileDataManager.loadDataListFiles2D(listValidImagesFiles, listValidMasksFiles)

print('Number Training images: %s' %(xTrain.shape[0]))
print('Number Validation images: %s' %(xValid.shape[0]))


# BUILDING MODEL
# ----------------------------------------------
print('_' * 30)
print('Building model...')
print('_' * 30)

model = DICTAVAILNETWORKS2D[IMODEL].getModel(IMAGES_HEIGHT, IMAGES_WIDTH)

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
    weightsPath = ModelsPath + '/bestModel.hdf5'
    model.load_weights(weightsPath)

starttime = time.time()

if( USE_DATAAUGMENTATION ):
    # Augmented Data generated "on the fly"
    datagen = Kpreprocessing.ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)

    model_info = model.fit_generator(datagen.flow(xTrain, yTrain,
                                                  batch_size=BATCH_SIZE),
                                     nb_epoch=NBEPOCHS,
                                     samples_per_epoch=xTrain.shape[0],
                                     verbose=1,
                                     validation_data=(xValid, yValid),
                                     callbacks=callbacks_list)
else:
    model_info = model.fit(xTrain, yTrain,
                           batch_size=BATCH_SIZE,
                           epochs=NBEPOCHS,
                           verbose=1,
                           shuffle=True,
                           #validation_split=0.2,
                           validation_data=(xValid, yValid),
                           callbacks=callbacks_list )

endtime = time.time()

print('Training finished in %f' %(endtime - starttime))
# ----------------------------------------------
