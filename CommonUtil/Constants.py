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
BASEDIR = '/home/antonio/Results/AirwaySegmen_DLCST/'


# ******************** INPUT IMAGES PARAMETERS ********************
# MUST BE MULTIPLES OF 16
# FOUND VERY CONVENIENT THE VALUES 36, 76, 146, ...
(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (224, 352, 240)

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

SHUFFLEIMAGES   = True
NORMALIZEDATA   = False
FORMATINOUTDATA = 'numpy_gzbi'

ISCLASSIFICATIONCASE = False

MULTICLASSCASE = False

NUMCLASSESMASKS = 2

if ISCLASSIFICATIONCASE:
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

NAME_IMAGES_TRAINING = ['images-03_img1', 'images-03_img2',
                        'images-04_img1', 'images-04_img2',
                        'images-05_img1', 'images-05_img2',
                        'images-08_img1', 'images-08_img2',
                        'images-09_img1', 'images-09_img2',
                        'images-10_img1', 'images-10_img2',
                        'images-11_img1', 'images-11_img2',
                        'images-14_img1', 'images-14_img2',
                        'images-16_img1', 'images-16_img2',
                        'images-19_img1', 'images-19_img2',
                        'images-23_img1', 'images-23_img2',
                        'images-25_img1', 'images-25_img2',
                        'images-27_img1', 'images-27_img2',
                        'images-28_img1', 'images-28_img2',
                        'images-30_img1', 'images-30_img2',
                        'images-31_img1', 'images-31_img2']
NAME_IMAGES_VALIDATION = ['images-01_img1', 'images-01_img2',
                          'images-02_img1', 'images-02_img2',
                          'images-17_img1', 'images-17_img2',
                          'images-20_img1', 'images-20_img2',
                          'images-21_img1', 'images-21_img2',
                          'images-24_img1', 'images-24_img2',
                          'images-29_img1', 'images-29_img2',
                          'images-32_img1', 'images-32_img2',]
NAME_IMAGES_TESTING = ['images-06_img1', 'images-06_img2',
                       'images-07_img1', 'images-07_img2',
                       'images-12_img1', 'images-12_img2',
                       'images-13_img1', 'images-13_img2',
                       'images-15_img1', 'images-15_img2',
                       'images-18_img1', 'images-18_img2',
                       'images-22_img1', 'images-22_img2',
                       'images-26_img1', 'images-26_img2']
# ******************** DATA DISTRIBUTION ********************


# ******************** PRE-PROCESSING PARAMETERS ********************
INVERTIMAGEAXIAL = False

MASKTOREGIONINTEREST = True

REDUCESIZEIMAGES = False

SIZEREDUCEDIMAGES = (256, 256)

CROPIMAGES = True

EXTENDSIZEIMAGES = False

CONSTRUCTINPUTDATADLCST = True

VOXELSBUFFERBORDER = (0, 0, 0, 0)
#VOXELSBUFFERBORDER = (30, 30, 0, 0)

CROPSIZEBOUNDINGBOX = (224, 352, 480)
#CROPSIZEBOUNDINGBOX = (352, 480)

CHECKBALANCECLASSES = True

CREATEIMAGESBATCHES = False

PROP_OVERLAP_Z_X_Y = (0.75, 0.0, 0.0)

VISUALPROCDATAINBATCHES = True
# ******************** PRE-PROCESSING PARAMETERS ********************


# ******************** TRAINING PARAMETERS ********************
NUM_EPOCHS  = 1000
BATCH_SIZE  = 1
IMODEL      = 'Unet3D'
IOPTIMIZER  = 'Adam'
ILOSSFUN    = 'DiceCoefficient'
LISTMETRICS =[]

NUM_FEATMAPS_FIRSTLAYER = 8

LEARN_RATE  = 1.0e-05

SLIDINGWINDOWIMAGES = False

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
