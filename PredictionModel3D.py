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

BATCH_SIZE = 1

IMODEL = 'Unet3D'

#MAIN
workDirsManager = WorkDirsManager(BASEDIR)
TestingDataPath = workDirsManager.getNameNewPath(workDirsManager.getNameTestingDataPath(), 'ProcVolsData')
ModelsPath      = workDirsManager.getNameModelsPath()


# LOADING DATA
# ----------------------------------------------
print('-' * 30)
print('Loading data...')
print('-' * 30)

listTestImagesFiles = sorted(glob(TestingDataPath + '/volsImages*.npy'))
listTestMasksFiles  = sorted(glob(TestingDataPath + '/volsMasks*.npy' ))

(xTest, yTest) = FileDataManager.loadDataListFiles3D(listTestImagesFiles, listTestMasksFiles)

print('Number testing images: %s' %(xTest.shape[0]))


# EVALUATE MODEL
# ----------------------------------------------
print('-' * 30)
print('Loading saved weights...')
print('-' * 30)

model = DICTAVAILNETWORKS3D[IMODEL].getModel(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH)
model.summary()

weightsPath = ModelsPath + '/bestModel.hdf5'
model.load_weights(weightsPath)

yPredict = model.predict(xTest, batch_size=1) #BATCH_SIZE

# Compute test accuracy
accuracy = DiceCoefficient.compute_home(yPredict, yTest)

print("Accuracy on Test Data: %0.2f" %(100*accuracy))