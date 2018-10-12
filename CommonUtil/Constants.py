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


DATADIR = '/home/antonio/Data/LUVAR_Raw/'
BASEDIR = '/home/antonio/Results/AirwaySegmen_LUVAR/'


# ******************** INPUT IMAGES PARAMETERS ********************
# MUST BE MULTIPLES OF 16
# FOUND VERY CONVENIENT THE VALUES 36, 76, 146, ...
#(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (104, 336, 224)
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
FORMATINOUTDATA = 'numpy_gzbi'
# ******************** INPUT IMAGES PARAMETERS ********************


# ******************** DATA DISTRIBUTION ********************
PROP_TRAINING   = 0.50
PROP_VALIDATION = 0.25
PROP_TESTING    = 0.25
DISTRIBUTE_RANDOM = False
# ******************** DATA DISTRIBUTION ********************


# ******************** PRE-PROCESSING PARAMETERS ********************
INVERTIMAGEAXIAL = False

MULTICLASSCASE = False

NUMCLASSESMASKS = 2

MASKTOREGIONINTEREST = True

REDUCESIZEIMAGES = False

SIZEREDUCEDIMAGES = (256, 256)

CROPIMAGES = True

EXTENDSIZEIMAGES = False

VOXELSBUFFERBORDER = (20, 0, 0, 0)

#CROPSIZEBOUNDINGBOX = (336, 448)
CROPSIZEBOUNDINGBOX = (352, 480)
#CROPSIZEBOUNDINGBOX = (448, 512)

CHECKBALANCECLASSES = True

CREATEIMAGESBATCHES = False

PROP_OVERLAP_Z_X_Y = (0.75, 0.0, 0.0)

VISUALPROCDATAINBATCHES = True
# ******************** PRE-PROCESSING PARAMETERS ********************


# ******************** TRAINING PARAMETERS ********************
NUM_EPOCHS  = 1000
BATCH_SIZE  = 1
IMODEL      = 'Unet3D_Dropout_BatchNormalization'
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

NUM_FEATMAPS_FIRSTLAYER = 16

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
ZOOM_RANGE = 0

ELASTICDEFORMATIONIMAGES = False

TYPEELASTICDEFORMATION = 'gridwise'

USETRANSFORMONVALIDATIONDATA = True

TYPEGPUINSTALLED = 'larger_GPU'

USEMULTITHREADING = False
# ******************** TRAINING PARAMETERS ********************


# ******************** RESTART PARAMETERS ********************
USE_RESTARTMODEL = False

RESTART_MODELFILE = 'lastEpoch'

RESTART_ONLY_WEIGHTS = False

EPOCH_RESTART = 40
# ******************** RESTART PARAMETERS ********************


# ******************** POST-PROCESSING PARAMETERS ********************
TYPEDATAPREDICT = 'testing'

PREDICTION_MODELFILE = 'lastEpoch'

PREDICTACCURACYMETRICS = 'DiceCoefficient'

LISTPOSTPROCESSMETRICS = ['DiceCoefficient',
                          'TruePositiveRate',
                          'TrueNegativeRate',
                          'FalsePositiveRate',
                          'FalseNegativeRate']

SAVEPREDICTMASKSLICES = True

CALCMASKSTHRESHOLDING = True

THRESHOLDVALUE = 0.5

ATTACHTRAQUEATOCALCMASKS = True

SAVETHRESHOLDIMAGES = True
# ******************** POST-PROCESSING PARAMETERS ********************
