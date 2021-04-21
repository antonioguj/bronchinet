#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=14G
#SBATCH -p hm
#SBATCH --gres=gpu:1
#SBATCH -t 3-00:00:00
#SBATCH -o ./Logs/out_%j.log
#SBATCH -e ./Logs/error_%j.log

# Load the modules
module purge
module load Python/3.7.4-GCCcore-8.3.0
module load libs/cuda/10.1.243
module load libs/cudnn/7.6.5.32-CUDA-10.1.243
#module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4

source /tmp/${SLURM_JOB_USER}.${SLURM_JOB_ID}/prolog.env

HOME="/trinity/home/agarcia/"
WORKDIR="${HOME}/Results/AirwaySegmentation_DLCST-LUVAR/"

export PYTHONPATH="${WORKDIR}/Code/:${PYTHONPATH}"

# Load virtual environment
source "${HOME}/Pyvenv-v.3.7.4/bin/activate"

# SETTINGS
MODELS_DIR="${WORKDIR}/Models/"
TRAINDATA_DIR="${WORKDIR}/TrainingData/"
VALIDDATA_DIR="${WORKDIR}/ValidationData/"
SIZE_IMAGES="(252,252,252)"
TYPE_NETWORK="UNet3DPlugin"
NET_NUM_FEATMAPS="16"
TYPE_LOSS="DiceCoefficient"
TYPE_OPTIMIZER="Adam"
LEARN_RATE="1.0e-04"
LIST_TYPE_METRICS="[]"
BATCH_SIZE="1"
NUM_EPOCHS="1000"
IS_VALID_CONVOLUTIONS="True"
IS_MASK_REGION_INTEREST="True"
IS_GENERATE_PATCHES="True"
TYPE_GENERATE_PATCHES="random_window"
PROP_OVERLAP_SLIDE_WINDOW="(0.25,0.0,0.0)"
NUM_RANDOM_PATCHES_EPOCH="8"
IS_TRANSFORM_IMAGES="True"
TYPE_TRANSFORM_IMAGES="rigid_trans"
TRANS_RIGID_ROTATION_RANGE="(10.0,7.0,7.0)"   # (plane_XY, plane_XZ, plane_YZ)
TRANS_RIGID_SHIFT_RANGE="(0.0,0.0,0.0)"       # (width, height, depth)
TRANS_RIGID_FLIP_DIRS="(True,True,True)"      # (horizontal, vertical, axialdir)
TRANS_RIGID_ZOOM_RANGE="0.25"                 # scale between (1 - val, 1 + val)
TRANS_RIGID_FILL_MODE="reflect"
FREQ_SAVE_CHECK_MODELS="2"
FREQ_VALIDATE_MODELS="2"
IS_USE_VALIDATION_DATA="True"
IS_SHUFFLE_TRAINDATA="True"
# --------

python3 "${WORKDIR}/Code/scripts_experiments/train_model.py" \
	--basedir=${WORKDIR} \
	--modelsdir=${MODELS_DIR} \
	--training_datadir=${TRAINDATA_DIR} \
	--validation_datadir=${VALIDDATA_DIR} \
	--size_in_images=${SIZE_IMAGES} \
	--type_network=${TYPE_NETWORK} \
	--net_num_featmaps=${NET_NUM_FEATMAPS} \
	--type_loss=${TYPE_LOSS} \
	--type_optimizer=${TYPE_OPTIMIZER} \
	--learn_rate=${LEARN_RATE} \
	--list_type_metrics=${LIST_TYPE_METRICS} \
	--batch_size=${BATCH_SIZE} \
	--num_epochs=${NUM_EPOCHS} \
	--is_valid_convolutions=${IS_VALID_CONVOLUTIONS} \
	--is_mask_region_interest=${IS_MASK_REGION_INTEREST} \
	--is_generate_patches=${IS_GENERATE_PATCHES} \
	--type_generate_patches=${TYPE_GENERATE_PATCHES} \
	--prop_overlap_slide_window=${PROP_OVERLAP_SLIDE_WINDOW} \
	--num_random_patches_epoch=${NUM_RANDOM_PATCHES_EPOCH} \
	--is_transform_images=${IS_TRANSFORM_IMAGES} \
	--type_transform_images=${TYPE_TRANSFORM_IMAGES} \
	--trans_rigid_rotation_range=${TRANS_RIGID_ROTATION_RANGE} \
	--trans_rigid_shift_range=${TRANS_RIGID_SHIFT_RANGE} \
	--trans_rigid_flip_dirs=${TRANS_RIGID_FLIP_DIRS} \
	--trans_rigid_zoom_range=${TRANS_RIGID_ZOOM_RANGE} \
	--trans_rigid_fill_mode=${TRANS_RIGID_FILL_MODE} \
	--freq_save_check_models=${FREQ_SAVE_CHECK_MODELS} \
	--freq_validate_models=${FREQ_VALIDATE_MODELS} \
	--is_use_validation_data=${IS_USE_VALIDATION_DATA} \
	--is_shuffle_traindata=${IS_SHUFFLE_TRAINDATA} #\
	# --is_restart="True" \
	# --in_config_file="${MODELS_DIR}/configparams.txt"
