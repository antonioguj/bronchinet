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


DATADIR = '/home/antonio/Data/LUVAR_Processed/'
#DATADIR = '/home/antonio/Data/DLCST_Processed/'
#DATADIR = '/home/antonio/Data/DLCST_Processed_ReferPechin/'
#DATADIR = '/home/antonio/Data/DLCST+LUVAR_Processed/'
#DATADIR = '/home/antonio/Data/EXACT_Processed/'

#BASEDIR = '/home/antonio/Results/AirwaySegmentation_LUVAR/'
BASEDIR = '/home/antonio/Results/AirwaySegmentation_LUVAR_Rescaled/'
#BASEDIR = '/home/antonio/Results/AirwaySegmentation_DLCST/'
#BASEDIR = '/home/antonio/Results/AirwaySegmentation_DLCST_RaghavPaper/'
#BASEDIR = '/home/antonio/Results/AirwaySegmentation_DLCST+LUVAR/'
#BASEDIR = '/home/antonio/Results/AirwaySegmentation_EXACT/'

TYPE_DNNLIBRARY_USED = 'Pytorch'
TYPEGPUINSTALLED     = 'larger_GPU'


# ******************** INPUT IMAGES PARAMETERS ********************
# MUST BE MULTIPLES OF 16
# FOUND VERY CONVENIENT THE VALUES 36, 76, 146, ...
#(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (240, 352, 240)
#(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (176, 352, 240)
(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (240, 240, 240)
#(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (160, 384, 272)

IMAGES_DIMS_X_Y   = (IMAGES_HEIGHT, IMAGES_WIDTH)
IMAGES_DIMS_Z_X_Y = (IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH)

FORMATINTDATA   = np.int16
FORMATSHORTDATA = np.uint8
FORMATREALDATA  = np.float32

FORMATIMAGESDATA     = FORMATINTDATA
FORMATMASKSDATA      = FORMATINTDATA
FORMATPHYSDISTDATA   = FORMATREALDATA
FORMATPROBABILITYDATA= FORMATREALDATA
FORMATFEATUREDATA    = FORMATREALDATA

SHUFFLETRAINDATA = True
NORMALIZEDATA    = False
ISBINARYTRAINMASKS = True

if ISBINARYTRAINMASKS:
    FORMATXDATA = FORMATIMAGESDATA
    FORMATYDATA = FORMATMASKSDATA
else:
    FORMATXDATA = FORMATIMAGESDATA
    FORMATYDATA = FORMATPHYSDISTDATA
# ******************** INPUT IMAGES PARAMETERS ********************


# ******************** DATA DISTRIBUTION ********************
PROP_DATA_TRAINING   = 0.50
PROP_DATA_VALIDATION = 0.15
PROP_DATA_TESTING    = 0.35

DISTRIBUTE_RANDOM = False
DISTRIBUTE_FIXED_NAMES = False
# ******************** DATA DISTRIBUTION ********************


# ******************** PRE-PROCESSING PARAMETERS ********************
MASKTOREGIONINTEREST = True

RESCALEIMAGES = False
ORDERINTERRESCALE = 3
#FIXEDRESCALERES = (0.6, 0.6, 0.6)
#FIXEDRESCALERES = (0.6, 0.55078125, 0.55078125)
#FIXEDRESCALERES = (0.6, 0.6, 0.6)
#FIXEDRESCALERES = (1.0, 0.78125, 0.78125)
FIXEDRESCALERES = None

CROPIMAGES = True
ISSAMEBOUNDBOXSIZEALLIMAGES = False
#CROPSIZEBOUNDINGBOX = (240, 352, 480)
FIXEDSIZEBOUNDINGBOX = (352, 480)
#FIXEDSIZEBOUNDINGBOX = (384, 544)
ISCALCBOUNDINGBOXINSLICES = False

SLIDINGWINDOWIMAGES = False
PROPOVERLAPSLIDINGWINDOW_Z_X_Y = (0.25, 0.0, 0.0)
#PROPOVERLAPSLIDINGWINDOW_Z_X_Y = (0.25, 0.25, 0.25)
#PROPOVERLAPSLIDINGWINDOW_Z_X_Y = (0.5, 0.5, 0.5)
PROPOVERLAPSLIDINGWINDOW_TESTING_Z_X_Y = (0.5, 0.5, 0.5)
#PROPOVERLAPSLIDINGWINDOW_TESTING_Z_X_Y = (0.5, 0.0, 0.0)

RANDOMCROPWINDOWIMAGES = True
NUMRANDOMIMAGESPERVOLUMEEPOCH = 8

TRANSFORMATIONRIGIDIMAGES = True
ROTATION_XY_RANGE = 10
ROTATION_XZ_RANGE = 5
ROTATION_YZ_RANGE = 5
HEIGHT_SHIFT_RANGE = 24
WIDTH_SHIFT_RANGE = 35
DEPTH_SHIFT_RANGE = 7
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
AXIALDIR_FLIP = True
ZOOM_RANGE = 0.25
FILL_MODE_TRANSFORM = 'reflect'

TRANSFORMELASTICDEFORMIMAGES = False
TYPETRANSFORMELASTICDEFORMATION = 'gridwise'

CREATEIMAGESBATCHES = False
SAVEVISUALIZEWORKDATA = False
# ******************** PRE-PROCESSING PARAMETERS ********************


# ******************** TRAINING PARAMETERS ********************
#IMODEL    	  = 'UnetGNN_OTF'
#IMODEL    	  = 'UnetGNN'
IMODEL       = 'Unet'
NUM_LAYERS   = 5
NUM_FEATMAPS = 16

TYPE_NETWORK         = 'classification'
TYPE_ACTIVATE_HIDDEN = 'relu'
TYPE_ACTIVATE_OUTPUT = 'sigmoid'
TYPE_PADDING_CONVOL  = 'same'
DISABLE_CONVOL_POOLING_LASTLAYER = False
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

NUMMAXTRAINIMAGES = 1
NUMMAXVALIDIMAGES = 1

ISVALIDCONVOLUTIONS = False
USEVALIDATIONDATA = True
FREQVALIDATEMODEL = 3
FREQSAVEINTERMODELS = 5
USETRANSFORMONVALIDATIONDATA = True

USEMULTITHREADING = False
WRITEOUTDESCMODELTEXT = False

# GNN-module parameters
ISTESTMODELSWITHGNN = False
SOURCEDIR_ADJS 	  = 'StoredAdjacencyMatrix/'
ISGNNWITHATTENTIONLAYS = False
# ******************** TRAINING PARAMETERS ********************


# ******************** RESTART PARAMETERS ********************
RESTART_MODEL = False
RESTART_EPOCH = 0
RESTART_ONLY_WEIGHTS = False
RESTART_FROMFILE = True
RESTART_FROMDIFFMODEL = False
# ******************** RESTART PARAMETERS ********************


# ******************** PREDICTION PARAMETERS ********************
PREDICTONRAWIMAGES = False
SAVEFEATMAPSLAYERS = False
NAMESAVEMODELLAYER = 'convU12'
FILTERPREDICTPROBMAPS = False
PROP_VALID_OUTUNET = 0.75
# ******************** PREDICTION PARAMETERS ********************


# ******************** POST-PROCESSING PARAMETERS ********************
LISTRESULTMETRICS = ['DiceCoefficient',
                     'AirwayCompleteness',
                     'AirwayVolumeLeakage',
                     'AirwayCompletenessModified',
                     'AirwayCentrelineLeakage',
                     'AirwayCentrelineFalsePositiveDistanceError',
                     'AirwayCentrelineFalseNegativeDistanceError']

THRESHOLDPOST = 0.5
ATTACHTRACHEAPREDICTION = False
REMOVETRACHEARESMETRICS = True

LISTMETRICSROCCURVE = ['DiceCoefficient',
                       'AirwayCompleteness',
                       'AirwayVolumeLeakage',
                       'AirwayCentrelineFalsePositiveDistanceError',
                       'AirwayCentrelineFalseNegativeDistanceError']
# ******************** POST-PROCESSING PARAMETERS ********************
