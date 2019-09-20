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


#DATADIR = '/home/antonio/Data/LUVAR_Processed/'
DATADIR = '/home/antonio/Data/DLCST_Processed/'
#DATADIR = '/home/antonio/Data/DLCST_Processed_ReferPechin/'

#BASEDIR = '/home/antonio/Results/AirwaySegmentation_LUVAR/'
BASEDIR = '/home/antonio/Results/AirwaySegmentation_DLCST/'
#BASEDIR = '/home/antonio/Results/AirwaySegmentation_UnetGNNs_DLCST/'
#BASEDIR = '/home/antonio/Results/AirwaySegmentation_DLCST_RaghavPaper/'

TYPE_DNNLIBRARY_USED = 'Pytorch'
TYPEGPUINSTALLED     = 'smaller_GPU'


# ******************** INPUT IMAGES PARAMETERS ********************
# MUST BE MULTIPLES OF 16
# FOUND VERY CONVENIENT THE VALUES 36, 76, 146, ...
#(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (240, 352, 240)
(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (176, 352, 240)
#(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (240, 256, 256)

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
PROP_DATA_VALIDATION = 0.125
PROP_DATA_TESTING    = 0.375

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

CROPIMAGES = True
#CROPSIZEBOUNDINGBOX = (240, 352, 480)
CROPSIZEBOUNDINGBOX = (352, 480)

RESCALEIMAGES = False
#FIXEDRESCALERES = (0.6, 0.6, 0.6)
FIXEDRESCALERES = (0.6, 0.55078125, 0.55078125)
EXTENDSIZEIMAGES = False

SLIDINGWINDOWIMAGES = True
SLIDEWIN_PROPOVERLAP_Z_X_Y = (0.25, 0.0, 0.0)
#SLIDEWIN_PROPOVERLAP_Z_X_Y = (0.5, 0.5, 0.5)

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

NUMMAXTRAINIMAGES = 16
NUMMAXVALIDIMAGES = 4

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
REMOVETRACHEARESMETRICS = True

LISTMETRICSROCCURVE = ['DiceCoefficient',
                       'AirwayCompleteness',
                       'AirwayVolumeLeakage',
                       'AirwayCentrelineFalsePositiveDistanceError',
                       'AirwayCentrelineFalseNegativeDistanceError']
# ******************** POST-PROCESSING PARAMETERS ********************
