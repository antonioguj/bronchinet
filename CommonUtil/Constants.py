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


DATADIR = '/home/antonio/Files_Project/testSegmentation/Data/LUVAR/'
BASEDIR = '/home/antonio/Files_Project/testSegmentation/Tests_LUVAR/'


# ******************** INPUT IMAGES PARAMETERS ********************
# MUST BE MULTIPLES OF 16
# FOUND VERY CONVENIENT THE VALUES 36, 76, 146, ...
(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (104, 352, 240)
#(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (76, 448, 256)

IMAGES_DIMS_X_Y   = (IMAGES_HEIGHT, IMAGES_WIDTH)
IMAGES_DIMS_Z_X_Y = (IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH)

IMAGES_SIZE_OUT_NNET = (IMAGES_DEPTHZ/4, 6*IMAGES_HEIGHT/10, 6*IMAGES_WIDTH/10)

FORMATIMAGEDATA   = np.int16
FORMATMASKDATA    = np.int8
FORMATPREDICTDATA = np.float32
FORMATPROPDATA    = np.float32

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

MULTICLASSCASE = False

NUMCLASSESMASKS = 2

CONFINEMASKSTOLUNGS = True

REDUCESIZEIMAGES = False

SIZEREDUCEDIMAGES = (256, 256)

CROPIMAGES = True

VOXELSBUFFERBORDER = (20, 0, 0, 0)

CROPSIZEBOUNDINGBOX = (352, 480)
#CROPSIZEBOUNDINGBOX = (448, 512)

CHECKBALANCECLASSES = True

CREATEIMAGESBATCHES = False

PROP_OVERLAP_Z_X_Y = (0.75, 0.0, 0.0)

SAVEVISUALPROCESSDATA = False
# ******************** PRE-PROCESSING PARAMETERS ********************


# ******************** TRAINING PARAMETERS ********************
NUM_EPOCHS  = 1000
BATCH_SIZE  = 1
IMODEL      = 'Unet3D_Tailored'
IOPTIMIZER  = 'Adam'
ILOSSFUN    = 'DiceCoefficient'
#ILOSSFUN    = 'CategoricalCrossEntropy'
LISTMETRICS =['BinaryCrossEntropy',
              'WeightedBinaryCrossEntropy',
              'DiceCoefficient',
              'TruePositiveRate',
              'TrueNegativeRate',
              'FalsePositiveRate',
              'FalseNegativeRate']
LEARN_RATE  = 1.0e-05

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

ELASTICDEFORMATIONIMAGES = False

TYPEELASTICDEFORMATION = 'gridwise'

USETRANSFORMONVALIDATIONDATA = True

USEMULTITHREADING = False
# ******************** TRAINING PARAMETERS ********************


# ******************** RESTART PARAMETERS ********************
USE_RESTARTMODEL = False

RESTART_MODELFILE = 'lastEpoch'

RESTART_ONLY_WEIGHTS = False

EPOCH_RESTART = 40
# ******************** RESTART PARAMETERS ********************


# ******************** POST-PROCESSING PARAMETERS ********************
PREDICTION_MODELFILE = 'lastEpoch'

PREDICTACCURACYMETRICS = 'DiceCoefficient'

LISTPOSTPROCESSMETRICS = ['DiceCoefficient',
                          'TruePositiveRate',
                          'TrueNegativeRate',
                          'FalsePositiveRate',
                          'FalseNegativeRate']

SAVEPREDICTMASKSLICES = True

THRESHOLDOUTPROBMAPS = False

THRESHOLDVALUE = 0.5

SAVETHRESHOLDIMAGES = True
# ******************** POST-PROCESSING PARAMETERS ********************
