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
import numpy as np


#MAIN

workDirsManager = WorkDirsManager(BASEDIR)
TestingDataPath = workDirsManager.getNameTestingDataPath()
ModelsPath      = workDirsManager.getNameModelsPath()


# LOADING DATA
# ----------------------------------------------
print('-' * 30)
print('Loading data...')
print('-' * 30)

( xTest, yTest ) = FileDataManager.loadDataFile2D(TestingDataPath, int(0.1*MAXIMAGESTOLOAD))


# EVALUATE MODEL
# ----------------------------------------------
print('-' * 30)
print('Loading saved weights...')
print('-' * 30)

model = Unet2D_Shallow.getModel(IMAGES_HEIGHT, IMAGES_WIDTH)

weightsPath = ModelsPath + 'bestModel.hdf5'
model.load_weights(weightsPath)

yPredict = model.predict(xTest)

ind_imgtest = np.random.randint(yTest.shape[0])

image_predict = yPredict[ind_imgtest].reshape(IMAGES_HEIGHT, IMAGES_WIDTH)
image_true    = yTest   [ind_imgtest].reshape(IMAGES_HEIGHT, IMAGES_WIDTH)

# Plot prediction and ground-truth to compare
PlotsManager.plot_compare_images(image_predict, image_true)

# Compute test accuracy
accuracy = Binary_CrossEntropy.compute_home(yPredict, yTest)

print("Accuracy on Test Data: %0.2f" %(100*accuracy))