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
IS_SLIDING_WINDOW_IMAGES="False"
PROP_OVERLAP_SLIDING_WINDOW="(0.25,0.0,0.0)"
IS_RANDOM_WINDOW_IMAGES="True"
NUM_RANDOM_PATCHES_EPOCH="8"
IS_TRANSFORM_RIGID_IMAGES="True"
IS_TRANSFORM_ELASTIC_IMAGES="False"
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
	--is_sliding_window_images=${IS_SLIDING_WINDOW_IMAGES} \
	--prop_overlap_sliding_window=${PROP_OVERLAP_SLIDING_WINDOW} \
	--is_random_window_images=${IS_RANDOM_WINDOW_IMAGES} \
	--num_random_patches_epoch=${NUM_RANDOM_PATCHES_EPOCH} \
	--is_transform_rigid_images=${IS_TRANSFORM_RIGID_IMAGES} \
	--is_transform_elastic_images=${IS_TRANSFORM_ELASTIC_IMAGES} #\
	# --is_restart="True" \
	# --in_config_file="${MODELS_DIR}/configparams.txt"
