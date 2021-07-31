#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH -p long
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH -o ./Logs/out_%j.log
#SBATCH -e ./Logs/error_%j.log

# Load the modules
module purge
module load Python/3.7.4-GCCcore-8.3.0
module load libs/cuda/10.1.243
module load libs/cudnn/7.6.5.32-CUDA-10.1.243
module load libs/tensorrt/6.0.1.5-CUDA-10.1.243
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4

source /tmp/${SLURM_JOB_USER}.${SLURM_JOB_ID}/prolog.env

HOME="/trinity/home/agarcia/"
WORKDIR="${HOME}/Results/MRIenhanceTests/"

export PYTHONPATH="${WORKDIR}/Code/:${PYTHONPATH}"

# Load virtual environment
source "${HOME}/Pyvenv-v.3.7.4/bin/activate"

# SETTINGS
MODELS_DIR="${WORKDIR}/Models_UNet5levs_DSSIM/"
TRAINDATA_DIR="${WORKDIR}/TrainingData/"
VALIDDATA_DIR="${WORKDIR}/ValidationData/"
SIZE_IMAGES="(288,288,96)"
TYPE_NETWORK="UNet3DPlugin5levels"
NET_NUM_FEATMAPS="16"
TYPE_LOSS="DSSIM"
#WEIGHT_COMBINED_LOSS="0.003"    # for DSSIM + L1
WEIGHT_COMBINED_LOSS="0.00006"  # for DSSIM + L2
#WEIGHT_COMBINED_LOSS="10.0"     # for Perceptual + L1
#WEIGHT_COMBINED_LOSS="0.2"      # for Perceptual + L2
TYPE_OPTIMIZER="Adam"
LEARN_RATE="1.0e-04"
LIST_TYPE_METRICS="[DSSIM]"
BATCH_SIZE="1"
NUM_EPOCHS="1000"
IS_VALID_CONVOLUTIONS="False"
IS_MASK_REGION_INTEREST="False"
IS_GENERATE_PATCHES="False"
TYPE_GENERATE_PATCHES="slide_window"
PROP_OVERLAP_SLIDE_WINDOW="(0.5,0.5,0.5)"
NUM_RANDOM_PATCHES_EPOCH="1"
IS_TRANSFORM_IMAGES="True"
TYPE_TRANSFORM_IMAGES="rigid_trans"
TRANS_RIGID_ROTATION_RANGE="(0.0,0.0,0.0)"    # (plane_XY, plane_XZ, plane_YZ)
TRANS_RIGID_SHIFT_RANGE="(0.0,0.0,0.0)"       # (width, height, depth)
TRANS_RIGID_FLIP_DIRS="(True,True,True)"      # (horizontal, vertical, axialdir)
TRANS_RIGID_ZOOM_RANGE="0.0"                  # scale between (1 - val, 1 + val)
TRANS_RIGID_FILL_MODE="nearest"
FREQ_SAVE_CHECK_MODELS="5"
FREQ_VALIDATE_MODELS="2"
IS_USE_VALIDATION_DATA="True"
IS_SHUFFLE_TRAINDATA="True"
LAYERS_VGG16_LOSS_PERCEPTUAL="[block1_conv1,block2_conv1,block3_conv1]"
WEIGHTS_VGG16_LOSS_PERCEPTUAL="[0.65,0.30,0.05]"
PROP_REDSIZE_VGG16_LOSS_PERCEPTUAL="0.0"
IS_LOAD_AUGMENTED_IMAGES_PER_LABEL="False"
NUM_AUGMENTED_IMAGES_PER_LABEL="28"
IS_TRAIN_NETWORK_2D="False"
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
	--weight_combined_loss=${WEIGHT_COMBINED_LOSS} \
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
	--is_shuffle_traindata=${IS_SHUFFLE_TRAINDATA} \
	--layers_vgg16_loss_perceptual=${LAYERS_VGG16_LOSS_PERCEPTUAL} \
	--weights_vgg16_loss_perceptual=${WEIGHTS_VGG16_LOSS_PERCEPTUAL} \
	--prop_redsize_vgg16_loss_perceptual=${PROP_REDSIZE_VGG16_LOSS_PERCEPTUAL} \
	--is_load_augmented_images_per_label=${IS_LOAD_AUGMENTED_IMAGES_PER_LABEL} \
	--num_augmented_images_per_label=${NUM_AUGMENTED_IMAGES_PER_LABEL} \
	--is_train_network_2D=${IS_TRAIN_NETWORK_2D} #\
	# --is_restart="True" \
	# --in_config_file="${MODELS_DIR}/configparams.txt" \
	# --is_restart_weights_diffmodel="False"
