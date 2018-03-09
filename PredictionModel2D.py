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
from CommonUtil.PlotsManager import *
from CommonUtil.WorkDirsManager import *
from Networks.Metrics import *
from Networks.Networks import *

BATCH_SIZE = 12

IMODEL = 'Unet2D'

#MAIN
workDirsManager = WorkDirsManager(BASEDIR)
TestingDataPath = workDirsManager.getNameTestingDataPath()
ModelsPath      = workDirsManager.getNameModelsPath()


# LOADING DATA
# ----------------------------------------------
print('-' * 30)
print('Loading data...')
print('-' * 30)

listTestImagesFiles = sorted(glob(TestingDataPath + 'images*.npy'))
listTestMasksFiles  = sorted(glob(TestingDataPath + 'masks*.npy' ))

(xTest, yTest) = FileDataManager.loadDataListFiles2D(listTestImagesFiles, listTestMasksFiles)

print('Number testing images: %s' %(xTest.shape[0]))


# EVALUATE MODEL
# ----------------------------------------------
print('-' * 30)
print('Loading saved weights...')
print('-' * 30)

model = DICTAVAILNETWORKS2D[IMODEL].getModel(IMAGES_HEIGHT, IMAGES_WIDTH)
model.summary()

weightsPath = ModelsPath + '/bestModel.hdf5'
model.load_weights(weightsPath)

yPredict = model.predict(xTest, batch_size=BATCH_SIZE)

# Plot prediction and ground-truth to compare
PlotsManager.plot_compare_images_2D(yPredict, yTest)

# Compute test accuracy
accuracy = DiceCoefficient.compute_home(yPredict, yTest)

print("Accuracy on Test Data: %0.2f" %(100*accuracy))