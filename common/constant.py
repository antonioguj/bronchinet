
import numpy as np
np.random.seed(2017)


FORMATINTDATA   = np.int16
FORMATSHORTDATA = np.uint8
FORMATREALDATA  = np.float32

FORMATIMAGESDATA     = FORMATINTDATA
FORMATMASKSDATA      = FORMATINTDATA
FORMATPHYSDISTDATA   = FORMATREALDATA
FORMATPROBABILITYDATA= FORMATREALDATA
FORMATFEATUREDATA    = FORMATREALDATA



# ******************** SET-UP WORKDIR ********************
DATADIR = '/home/antonio/Data/DLCST_Testing/'
BASEDIR = '/home/antonio/Results/AirwaySegmentation_DLCST/'

# Names input and output directories
NAME_RAWIMAGES_RELPATH        = 'Images/'
NAME_RAWLABELS_RELPATH        = 'Airways/'
NAME_RAWROIMASKS_RELPATH      = 'Lungs/'
NAME_RAWCOARSEAIRWAYS_RELPATH = 'CoarseAirways/'
NAME_RAWCENTRELINES_RELPATH   = 'Centrelines/'
NAME_RAWEXTRALABELS_RELPATH   = 'Centrelines/'
NAME_REFERKEYS_RELPATH        = 'Images/'
NAME_PROCIMAGES_RELPATH       = 'ImagesWorkData/'
NAME_PROCLABELS_RELPATH       = 'LabelsWorkData/'
NAME_PROCEXTRALABELS_RELPATH  = 'ExtraLabelsWorkData/'
NAME_CROPBOUNDINGBOX_FILE     = 'cropBoundingBoxes_images.npy'
NAME_RESCALEFACTOR_FILE       = 'rescaleFactors_images.npy'
NAME_REFERKEYSPROCIMAGE_FILE  = 'referenceKeys_procimages.npy'
NAME_TRAININGDATA_RELPATH     = 'TrainingData/'
NAME_VALIDATIONDATA_RELPATH   = 'ValidationData/'
NAME_TESTINGDATA_RELPATH      = 'TestingData/'
NAME_CONFIGPARAMS_FILE        = 'configparams.txt'
NAME_DESCMODEL_LOGFILE        = 'logdescmodel.txt'
NAME_TRAINDATA_LOGFILE        = 'traindatafiles.txt'
NAME_VALIDDATA_LOGFILE        = 'validdatafiles.txt'
NAME_LOSSHISTORY_FILE         = 'lossHistory.csv'
NAME_SAVEDMODEL_INTER_KERAS   = 'model_e{epoch:02d}.hdf5'
NAME_SAVEDMODEL_LAST_KERAS    = 'model_last.hdf5'
NAME_SAVEDMODEL_EPOCH_TORCH   = 'model_e%0.2d.pt'
NAME_SAVEDMODEL_LAST_TORCH    = 'model_last.pt'
NAME_TEMPOPOSTERIORS_RELPATH  = 'Predictions/PosteriorsWorkData/'
NAME_POSTERIORS_RELPATH       = 'Predictions/Posteriors/'
NAME_PREDBINARYMASKS_RELPATH  = 'Predictions/BinaryMasks/'
NAME_REFERKEYSPOSTERIORS_FILE = 'Predictions/referenceKeys_posteriors.npy'
# ******************** SET-UP WORKDIR ********************


# ******************** DATA DISTRIBUTION ********************
#PROPDATA_TRAINVALIDTEST = (0.84, 0.16, 0.0) # for EXACT
#PROPDATA_TRAINVALIDTEST = (0.5, 0.13, 0.37) # for DLCST+LUVAR
PROPDATA_TRAINVALIDTEST = (0.5, 0.12, 0.38)
# ******************** DATA DISTRIBUTION ********************


# ******************** PREPROCESS PARAMETERS ********************
SHUFFLETRAINDATA = True
NORMALIZEDATA    = False
ISBINARYTRAINMASKS = True

if ISBINARYTRAINMASKS:
    FORMAT_XDATA = FORMATIMAGESDATA
    FORMAT_YDATA = FORMATMASKSDATA
else:
    FORMAT_XDATA = FORMATIMAGESDATA
    FORMAT_YDATA = FORMATPHYSDISTDATA

IS_MASK_REGION_INTEREST = True

RESCALEIMAGES = False
#FIXEDRESCALERES = (0.6, 0.55, 0.55)   # for LUVAR
FIXEDRESCALERES = (0.8, 0.69, 0.69)   # for EXACT
#FIXEDRESCALERES = None

CROPIMAGES = True
ISTWOBOUNDINGBOXEACHLUNGS = False
SIZEBUFFERBOUNDBOXBORDERS = (20, 20, 20)
ISSAMESIZEBOUNDBOXALLIMAGES = False
ISCALCBOUNDINGBOXINSLICES = False
FIXEDSIZEBOUNDINGBOX = None
#FIXEDSIZEBOUNDINGBOX = (352, 480)
#FIXEDSIZEBOUNDINGBOX = (332, 316, 236) # for DLCST and full size lungs
#FIXEDSIZEBOUNDINGBOX = (508, 332, 236) # for EXACT and full size lungs

SLIDINGWINDOWIMAGES = False
PROPOVERLAPSLIDINGWINDOW_Z_X_Y = (0.25, 0.0, 0.0)
PROPOVERLAPSLIDINGWINDOW_TESTING_Z_X_Y = (0.5, 0.5, 0.5)

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
# ******************** PREPROCESS PARAMETERS ********************


# ******************** TRAINING PARAMETERS ********************
TYPE_DNNLIB_USED = 'Pytorch'
#TYPEGPUINSTALLED = 'larger_GPU'

#(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (176, 352, 240)
#(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (256, 256, 256) # for Non-valid convolutions
(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (252, 252, 252) # for Valid convolutions
#(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (332, 316, 236) # for DLCST and full size lungs
#(IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH) = (508, 332, 236) # for EXACT and full size lungs
IMAGES_DIMS_Z_X_Y = (IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH)

TYPE_NETWORK = 'Unet3D_Plugin'
NUM_LEVELS   = 5
NUM_FEATMAPS = 16

ISMODEL_IN_GPU       = True
ISMODEL_HALFPRECISION = False
TYPE_ACTIVATE_HIDDEN = 'relu'
TYPE_ACTIVATE_OUTPUT = 'sigmoid'
TYPE_PADDING_CONVOL  = 'same'
DISABLE_CONVOL_POOLING_LASTLAYER = False
ISUSE_DROPOUT        = False
ISUSE_BATCHNORMALIZE = False
TAILORED_BUILD_MODEL = True

NUM_EPOCHS = 1000
BATCH_SIZE = 1
TYPE_OPTIMIZER = 'Adam'
LEARN_RATE = 1.0e-04
TYPE_LOSS   = 'DiceCoefficient'
LIST_TYPE_METRICS = []
#LISTMETRICS = ['DiceCoefficient', 'TruePositiveRate', 'FalsePositiveRate', 'TrueNegativeRate', 'FalseNegativeRate']

NUMMAXTRAINIMAGES = 50
NUMMAXVALIDIMAGES = 20

IS_VALID_CONVOLUTIONS = True
USEVALIDATIONDATA = True
FREQ_VALIDATE_MODEL = 3
FREQ_SAVE_INTER_MODELS = 5
USE_TRANSFORM_VALIDATION_DATA = True

USEMULTITHREADING = False
WRITE_OUT_DESC_MODEL_TEXT = False

# GNN-module parameters
ISTESTMODELSWITHGNN = False
SOURCEDIR_ADJS 	  = 'StoredAdjacencyMatrix/'
ISGNNWITHATTENTIONLAYS = False
# ******************** TRAINING PARAMETERS ********************


# ******************** RESTART PARAMETERS ********************
IS_RESTART_MODEL = False
IS_RESTART_ONLY_WEIGHTS = False
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
                     'AirwayCentrelineLeakage',
                     'AirwayCentrelineFalsePositiveDistanceError',
                     'AirwayCentrelineFalseNegativeDistanceError']

THRESHOLDPOST = 0.5
ATTACHCOARSEAIRWAYSMASK = True
REMOVETRACHEACALCMETRICS = True

LISTMETRICSROCCURVE = ['DiceCoefficient',
                       'AirwayCompleteness',
                       'AirwayVolumeLeakage',
                       'AirwayCentrelineFalsePositiveDistanceError',
                       'AirwayCentrelineFalseNegativeDistanceError']
# ******************** POST-PROCESSING PARAMETERS ********************
