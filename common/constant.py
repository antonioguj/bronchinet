
import numpy as np
np.random.seed(2017)

DATADIR = '/home/antonio/Data/EXACT_Testing/'
BASEDIR = '//home/antonio/Results/AirwaySegmentation_EXACT/'


# NAMES INPUT / OUTPUT DIR
NAME_RAW_IMAGES_RELPATH         = 'Images/'
NAME_RAW_LABELS_RELPATH         = 'Airways/'
NAME_RAW_ROIMASKS_RELPATH       = 'Lungs/'
NAME_RAW_COARSEAIRWAYS_RELPATH  = 'CoarseAirways/'
NAME_RAW_CENTRELINES_RELPATH    = 'Centrelines/'
NAME_RAW_EXTRALABELS_RELPATH    = 'Centrelines/'
NAME_REFERENCE_FILES_RELPATH    = 'Images/'
NAME_PROC_IMAGES_RELPATH        = 'ImagesWorkData/'
NAME_PROC_LABELS_RELPATH        = 'LabelsWorkData/'
NAME_PROC_EXTRA_LABELS_RELPATH  = 'ExtraLabelsWorkData/'
NAME_CROP_BOUNDINGBOX_FILE      = 'cropBoundingBoxes_images.npy'
NAME_RESCALE_FACTOR_FILE        = 'rescaleFactors_images.npy'
NAME_REFERENCE_KEYS_PROCIMAGE_FILE = 'referenceKeys_procimages.npy'
NAME_TRAININGDATA_RELPATH       = 'TrainingData/'
NAME_VALIDATIONDATA_RELPATH     = 'ValidationData/'
NAME_TESTINGDATA_RELPATH        = 'TestingData/'
NAME_CONFIG_PARAMS_FILE         = 'configparams.txt'
NAME_DESCRIPT_MODEL_LOGFILE     = 'descriptmodel.txt'
NAME_TRAINDATA_LOGFILE          = 'traindatafiles.txt'
NAME_VALIDDATA_LOGFILE          = 'validdatafiles.txt'
NAME_LOSSHISTORY_FILE           = 'losshistory.csv'
NAME_SAVEDMODEL_INTER_KERAS     = 'model_e{epoch:02d}.hdf5'
NAME_SAVEDMODEL_LAST_KERAS      = 'model_last.hdf5'
NAME_SAVEDMODEL_EPOCH_TORCH     = 'model_e%0.2d.pt'
NAME_SAVEDMODEL_LAST_TORCH      = 'model_last.pt'
NAME_TEMPO_POSTERIORS_RELPATH   = 'Predictions/PosteriorsWorkData/'
NAME_POSTERIORS_RELPATH         = 'Predictions/Posteriors/'
NAME_PRED_BINARYMASKS_RELPATH   = 'Predictions/BinaryMasks/'
NAME_PRED_CENTRELINES_RELPATH   = 'Predictions/Centrelines/'
NAME_REFERENCE_KEYS_POSTERIORS_FILE = 'Predictions/referenceKeys_posteriors.npy'
NAME_PRED_RESULT_METRICS_FILE   = 'Predictions/result_metrics.csv'


# PREPROCESSING
IS_SHUFFLE_TRAINDATA            = True
IS_NORMALIZE_DATA               = False
IS_BINARY_TRAIN_MASKS           = True
IS_MASK_REGION_INTEREST         = True
IS_RESCALE_IMAGES               = False
#FIXED_RESCALE_RESOL             = (0.6, 0.55, 0.55)   # for LUVAR
FIXED_RESCALE_RESOL             = (0.8, 0.69, 0.69)   # for EXACT
#FIXED_RESCALE_RESOL             = None
IS_CROP_IMAGES                  = True
IS_TWO_BOUNDBOXES_EACH_LUNGS    = False
SIZE_BUFFER_BOUNDBOX_BORDERS    = (20, 20, 20)
IS_SAME_SIZE_BOUNDBOX_ALL_IMAGES= False
IS_CALC_BOUNDINGBOX_IN_SLICES   = False
FIXED_SIZE_BOUNDING_BOX         = None
#FIXED_SIZE_BOUNDING_BOX         = (352, 480)
#FIXED_SIZE_BOUNDING_BOX         = (332, 316, 236) # for DLCST and full size lungs
#FIXED_SIZE_BOUNDING_BOX         = (508, 332, 236) # for EXACT and full size lungs


# DATA AUGMENTATION IN TRAINING
USE_SLIDING_WINDOW_IMAGES       = False
PROP_OVERLAP_SLIDING_WINDOW     = (0.25, 0.0, 0.0)
USE_RANDOM_WINDOW_IMAGES        = True
NUM_RANDOM_PATCHES_EPOCH        = 8
USE_TRANSFORM_RIGID_IMAGES      = True
USE_TRANSFORM_ELASTICDEFORM_IMAGES = True
TRANS_ROTATION_XY_RANGE         = 0 # 10
TRANS_ROTATION_XZ_RANGE         = 0 # 5
TRANS_ROTATION_YZ_RANGE         = 0 # 5
TRANS_HEIGHT_SHIFT_RANGE        = 0 # 24
TRANS_WIDTH_SHIFT_RANGE         = 0 # 35
TRANS_DEPTH_SHIFT_RANGE         = 0 # 7
TRANS_HORIZONTAL_FLIP           = True
TRANS_VERTICAL_FLIP             = True
TRANS_AXIALDIR_FLIP             = True
TRANS_ZOOM_RANGE                = 0.25
TRANS_FILL_MODE_TRANSFORM       = 'reflect'
TYPE_TRANSFORM_ELASTICDEFORM_IMAGES = 'gridwise'


# DISTRIBUTE DATA TRAIN / VALID / TEST
DIST_PROPDATA_TRAINVALIDTEST    = (0.5, 0.12, 0.38)
#DIST_PROPDATA_TRAINVALIDTEST    = (0.84, 0.16, 0.0) # for EXACT
#DIST_PROPDATA_TRAINVALIDTEST    = (0.5, 0.13, 0.37) # for DLCST+LUVAR


# TRAINING MODELS
TYPE_DNNLIB_USED            = 'Pytorch'
#TYPE_GPU_USED               = 'larger_GPU'
IS_MODEL_IN_GPU             = True
IS_MODEL_HALF_PRECISION     = False
USE_MULTITHREADING          = False
#SIZE_IN_IMAGES              = (176, 352, 240)
#SIZE_IN_IMAGES              = (256, 256, 256) # for Non-valid convolutions
SIZE_IN_IMAGES              = (252, 252, 252) # for Valid convolutions
#SIZE_IN_IMAGES              = (332, 316, 236) # for DLCST and full size lungs
#SIZE_IN_IMAGES              = (508, 332, 236) # for EXACT and full size lungs
MAX_TRAIN_IMAGES            = 50
MAX_VALID_IMAGES            = 20
BATCH_SIZE                  = 1
NUM_EPOCHS                  = 1000
TYPE_NETWORK                = 'UNet3D_Plugin'
NET_NUM_LEVELS              = 5
NET_NUM_FEATMAPS            = 10
TYPE_ACTIVATE_HIDDEN        = 'relu'
TYPE_ACTIVATE_OUTPUT        = 'sigmoid'
IS_DISABLE_CONVOL_POOLING_LASTLAYER = False
NET_USE_DROPOUT             = False
NET_USE_BATCHNORMALIZE      = False
TYPE_OPTIMIZER              = 'Adam'
LEARN_RATE                  = 1.0e-04
TYPE_LOSS                   = 'DiceCoefficient'
LIST_TYPE_METRICS           = []
#LIST_TYPE_METRICS           = ['TruePositiveRate', 'FalsePositiveRate', 'TrueNegativeRate', 'FalseNegativeRate']
IS_VALID_CONVOLUTIONS       = True
USE_VALIDATION_DATA         = True
FREQ_VALIDATE_MODEL         = 3
FREQ_SAVE_INTER_MODELS      = 5
USE_TRANSFORM_VALIDATION_DATA = True
WRITE_OUT_DESC_MODEL_TEXT   = False
IS_RESTART_MODEL            = False
IS_RESTART_ONLY_WEIGHTS     = False
# GNN-module parameters
USE_MODELS_WITH_GNN         = False
ADJACENCY_GNN_STOREDIR      = 'StoredAdjacencyMatrix/'
IS_GNN_WITH_ATTENTION       = False


# PREDICTIONS / POST-PROCESSING PARAMETERS
PROP_OVERLAP_SLIDING_WINDOW_TESTING = (0.5, 0.5, 0.5)
IS_FILTER_PRED_PROBMAPS     = False
PROP_VALID_OUTPUT_NNET      = 0.75
POST_THRESHOLD_VALUE        = 0.5
IS_ATTACH_COARSE_AIRWAYS    = True
IS_REMOVE_TRACHEA_CALC_METRICS = True
LIST_TYPE_METRICS_RESULT    = ['DiceCoefficient',
                               'AirwayCompleteness',
                               'AirwayVolumeLeakage',
                               'AirwayCentrelineLeakage',
                               'AirwayCentrelineDistanceFalsePositiveError',
                               'AirwayCentrelineDistanceFalseNegativeError']
LIST_TYPE_METRICS_ROC_CURVE = ['DiceCoefficient',
                               'AirwayCompleteness',
                               'AirwayVolumeLeakage',
                               'AirwayCentrelineDistanceFalsePositiveError',
                               'AirwayCentrelineDistanceFalseNegativeError']
METRIC_EVALUATE_THRESHOLD   = 'AirwayVolumeLeakage'
IS_SAVE_FEATMAPS_LAYERS     = False
NAME_SAVE_FEATS_MODEL_LAYER = 'convU12'
