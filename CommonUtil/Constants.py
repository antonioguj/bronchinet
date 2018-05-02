#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

import numpy as np
np.random.seed(2017)


DATADIR = '/home/antonio/testSegmentation/Data/LUVAR/'
BASEDIR = '/home/antonio/testSegmentation/Tests_LUVAR/'


# ******************** INPUT IMAGES PARAMETERS ********************
# MUST BE MULTIPLES OF 16
# FOUND VERY CONVENIENT THE VALUES 36, 76, 146, ...
IMAGES_DEPTHZ = 36
IMAGES_HEIGHT = 352
#IMAGES_HEIGHT = 256
IMAGES_WIDTH  = 240
#IMAGES_WIDTH  = 256

IMAGES_DIMS_X_Y   = (IMAGES_HEIGHT, IMAGES_WIDTH)
IMAGES_DIMS_Z_X_Y = (IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH)

FORMATIMAGEDATA   = np.int16
FORMATMASKDATA    = np.int8
FORMATPREDICTDATA = np.float32

SHUFFLEIMAGES   = True
NORMALIZEDATA   = False
FORMATINOUTDATA = 'numpy'
# ******************** INPUT IMAGES PARAMETERS ********************


# ******************** DATA DISTRIBUTION ********************
PROP_TRAINING   = 0.50
PROP_VALIDATION = 0.25
PROP_TESTING    = 0.25
DISTRIBUTE_RANDOM = False
# ******************** DATA DISTRIBUTION ********************


# ******************** PRE-PROCESSING PARAMETERS ********************
TYPEDATA = 'training'

REDUCESIZEIMAGES = False

SIZEREDUCEDIMAGES = (256, 256)

CROPIMAGES = True

CROPSIZEBOUNDINGBOX = (352, 480)

MULTICLASSCASE = False

NUMCLASSESMASKS = 2

CONFINEMASKSTOLUNGS = True

CHECKBALANCECLASSES = True

CREATEIMAGESBATCHES = False

PROP_OVERLAP_Z_X_Y = (0.75, 0.0, 0.0)

SAVEVISUALPROCESSDATA = False
# ******************** PRE-PROCESSING PARAMETERS ********************


# ******************** TRAINING PARAMETERS ********************
NUM_EPOCHS = 1000
BATCH_SIZE = 1
IMODEL     = 'Unet3D_Shallow_Tailored'
IOPTIMIZER = 'Adam'
ILOSSFUN   = 'WeightedBinaryCrossEntropy_Masked'
#ILOSSFUN   = 'CategoricalCrossEntropy'
IMETRICS   = 'DiceCoefficient_Masked'
#IMETRICS   = 'DiceCoefficient'
LEARN_RATE = 1.0e-05

SLIDINGWINDOWIMAGES = True

TRANSFORMATIONIMAGES = True

ROTATION_XY_RANGE = 10
ROTATION_XZ_RANGE = 5
ROTATION_YZ_RANGE = 5
HEIGHT_SHIFT_RANGE = 24
WIDTH_SHIFT_RANGE = 35
DEPTH_SHIFT_RANGE = 7
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
DEPTHZ_FLIP = True

USE_RESTARTMODEL = False

EPOCH_RESTART = 40
RESTART_MODELFILE = 'lastEpoch'
# ******************** TRAINING PARAMETERS ********************


# ******************** POST-PROCESSING PARAMETERS ********************
PREDICTION_MODELFILE = 'lastEpoch'

PREDICTACCURACYMETRICS = 'DiceCoefficient_Masked'
#PREDICTACCURACYMETRICS = 'DiceCoefficient'

THRESHOLDOUTIMAGES = False

THRESHOLDVALUE = 0.5

SAVEVISUALPREDICTDATA = False

SAVEPREDICTIONIMAGES = True
# ******************** POST-PROCESSING PARAMETERS ********************