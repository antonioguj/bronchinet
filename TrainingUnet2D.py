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
from keras import callbacks as Kcallbacks
from keras import optimizers as Koptimizers
from keras.preprocessing import image as Kpreprocessing
import datetime
import sys

USE_DATAAUGMENTATION = False

USE_RESTARTMODEL = False


#MAIN

workDirsManager    = WorkDirsManager(BASEDIR)
TrainingDataPath   = workDirsManager.getNameTrainingDataPath()
ValidationDataPath = workDirsManager.getNameValidationDataPath()
TestingDataPath    = workDirsManager.getNameTestingDataPath()
ModelsPath         = workDirsManager.getNameModelsPath()



# LOADING DATA
# ----------------------------------------------
print('-' * 30)
print('Loading data...')
print('-' * 30)

( xTrain, yTrain ) = FileDataManager.loadDataFile2D(TrainingDataPath,   MAXIMAGESTOLOAD)
( xValid, yValid ) = FileDataManager.loadDataFile2D(ValidationDataPath, int(MAXIMAGESTOLOAD*0.1))


# BUILDING MODEL
# ----------------------------------------------
print('_' * 30)
print('Building model...')
print('_' * 30)

optimizer    = Koptimizers.Adam( lr=1.0e-03 )
#lossfunction = LossFunction(Binary_CrossEntropy).compute
lossfunction = 'binary_crossentropy'
metrics      = []

model = Unet2D_Shallow.getModel(IMAGES_HEIGHT, IMAGES_WIDTH)

# Compile model
model.compile(optimizer=optimizer,
              loss=lossfunction,
              metrics=metrics)
model.summary()

# Callbacks:
callbacks_list = []
callbacks_list.append(RecordLossHistory(ModelsPath))

filename = ModelsPath + 'best_weights.{epoch:02d}-{val_loss:.5f}.hdf5'
callbacks_list.append(callbacks.ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True))

callbacks_list.append(Kcallbacks.EarlyStopping(monitor='val_loss', patience=10, mode='max'))
callbacks_list.append(Kcallbacks.TerminateOnNaN())
# ----------------------------------------------


# TRAINING MODEL
# ----------------------------------------------
print('-' * 30)
print('Training model...')
print('-' * 30)

if USE_RESTARTMODEL:
    weightsPath = ModelsPath + 'bestModel.hdf5'
    model.load_weights(weightsPath)

starttime = datetime.datetime.now()

batch_size = 4
nbepochs   = 500

if( USE_DATAAUGMENTATION ):
    # Augmented Data generated "on the fly"
    datagen = Kpreprocessing.ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)

    model_info = model.fit_generator(datagen.flow(xTrain, yTrain, batch_size=batch_size),
                                     nb_epoch=nbepochs,
                                     samples_per_epoch=xTrain.shape[0],
                                     verbose=1,
                                     validation_data=(xValid, yValid),
                                     callbacks=callbacks_list)
else:
    model_info = model.fit(xTrain, yTrain,
                           batch_size=batch_size,
                           epochs=nbepochs,
                           verbose=1,
                           shuffle=True,
                           validation_data=(xValid, yValid),
                           callbacks=callbacks_list )

endtime = datetime.datetime.now()

print('Training finished in %f' %(endtime - starttime))
# ----------------------------------------------