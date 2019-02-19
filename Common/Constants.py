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


DATADIR = '/home/antonio/Data/DLCST_Raw/'
BASEDIR = '/home/antonio/Results/AirwaySegmentation_DLCST/'

TYPE_DNNLIBRARY_USED = 'Pytorch'


# ******************** INPUT IMAGES PARAMETERS ********************
# MUST BE MULTIPLES OF 16
# FOUND VERY CONVENIENT THE VALUES 36, 76, 146, ...
#(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (240, 352, 240)
(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (240, 352, 240)
#(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (224, 256, 256)

IMAGES_DIMS_X_Y   = (IMAGES_HEIGHT, IMAGES_WIDTH)
IMAGES_DIMS_Z_X_Y = (IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH)

FORMATINTDATA   = np.int16
FORMATSHORTDATA = np.int16
FORMATREALDATA  = np.float32

FORMATIMAGESDATA     = FORMATINTDATA
FORMATMASKSDATA      = FORMATSHORTDATA
FORMATPHYSDISTDATA   = FORMATREALDATA
FORMATPROBABILITYDATA= FORMATREALDATA
FORMATFEATUREDATA    = FORMATREALDATA

SHUFFLETRAINDATA = True
NORMALIZEDATA    = False
FORMATTRAINDATA  = 'numpy_gzbi'

ISCLASSIFICATIONDATA = True

if ISCLASSIFICATIONDATA:
    FORMATXDATA = FORMATIMAGESDATA
    FORMATYDATA = FORMATMASKSDATA
else:
    FORMATXDATA = FORMATIMAGESDATA
    FORMATYDATA = FORMATPHYSDISTDATA
# ******************** INPUT IMAGES PARAMETERS ********************


# ******************** DATA DISTRIBUTION ********************
PROP_DATA_TRAINING   = 0.50
PROP_DATA_VALIDATION = 0.25
PROP_DATA_TESTING    = 0.25

DISTRIBUTE_RANDOM = False
DISTRIBUTE_FIXED_NAMES = False

#for LUVAR Data
# NAME_IMAGES_TRAINING = ['images-01', 'images-02', 'images-03', 'images-06', 'images-08', 'images-10',
#                         'images-11', 'images-12', 'images-14', 'images-15', 'images-16', 'images-18']
# NAME_IMAGES_VALIDATION = ['images-05', 'images-09', 'images-13', 'images-17', 'images-19', 'images-20']
# NAME_IMAGES_TESTING = ['images-04', 'images-07', 'images-21', 'images-22', 'images-23', 'images-24']
# ******************** DATA DISTRIBUTION ********************


# ******************** PRE-PROCESSING PARAMETERS ********************
MASKTOREGIONINTEREST = True

RESCALEIMAGES = True

FIXEDRESCALERES = (0.6, 0.6, 0.6)

CROPIMAGES = True

#CROPSIZEBOUNDINGBOX = (240, 352, 480)
CROPSIZEBOUNDINGBOX = (352, 480)

EXTENDSIZEIMAGES = False

CREATEIMAGESBATCHES = False

PROP_OVERLAP_Z_X_Y = (0.75, 0.0, 0.0)

SAVEVISUALIZEPROCDATA = False
# ******************** PRE-PROCESSING PARAMETERS ********************


# ******************** TRAINING PARAMETERS ********************
NUM_LAYERS           = 5
NUM_FEATMAPS_BASE    = 8
TYPE_NETWORK         = 'classification'
TYPE_ACTIVATE_HIDDEN = 'relu'
TYPE_ACTIVATE_OUTPUT = 'sigmoid'
TYPE_PADDING_CONVOL  = 'same'
DISABLE_CONVOL_POOLING_LASTLAYER = True
ISUSE_DROPOUT        = False
ISUSE_BATCHNORMALIZE = False
TAILORED_BUILD_MODEL = True

NUM_EPOCHS = 1000
BATCH_SIZE = 1
IOPTIMIZER = 'Adam'
LEARN_RATE = 1.0e-04
ILOSSFUN   = 'DiceCoefficient'
LISTMETRICS = []
# LISTMETRICS = ['BinaryCrossEntropy',
#                'WeightedBinaryCrossEntropy',
#                'DiceCoefficient',
#                'TruePositiveRate',
#                'FalsePositiveRate',
#                'TrueNegativeRate',
#                'FalseNegativeRate']

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
ZOOM_RANGE = 0.0

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

FILTERPREDICTPROBMAPS = True

PROP_VALID_OUTUNET = 0.75

SAVEFEATMAPSLAYERS = False

NAMESAVEMODELLAYER = 'conv3d_18'

SAVEPREDICTMASKSLICES = False

CALCMASKSTHRESHOLDING = True

THRESHOLDVALUE = 0.5

ATTACHTRAQUEATOCALCMASKS = False

SAVETHRESHOLDIMAGES = True
# ******************** POST-PROCESSING PARAMETERS ********************
