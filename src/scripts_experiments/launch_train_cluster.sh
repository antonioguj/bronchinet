#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH -p hm
#SBATCH --gres=gpu:1
#SBATCH -t 3-00:00:00
#SBATCH -o ./Logfiles/out_%j.log
#SBATCH -e ./Logfiles/error_%j.log

# Load the modules
module purge
module load Python/3.7.4-GCCcore-8.3.0
module load libs/cuda/10.1.243
module load libs/cudnn/7.6.5.32-CUDA-10.1.243
#module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4

source /tmp/${SLURM_JOB_USER}.${SLURM_JOB_ID}/prolog.env


HOME="/trinity/home/agarcia/"
WORKDIR="${HOME}/Results/MRIenhanceTests/"

export PYTHONPATH="${WORKDIR}/Code/:${PYTHONPATH}"

# Load virtual environment
source "${HOME}/Pyvenv-v.3.7.4/bin/activate"


# ---------- SETTINGS ----------
#
MODELS_DIR="${WORKDIR}/Models_UNet5levs_RestartNoSkipConn_Perceptual_RigidTransform/"
TRAINDATA_DIR="${WORKDIR}/TrainingData_NotMasked/"
VALIDDATA_DIR="${WORKDIR}/ValidationData_NotMasked/"
SIZE_IMAGES="(288,288,96)"
TYPE_NETWORK="UNet3D_Plugin5levels"
NET_NUM_FEATMAPS="16"
TYPE_LOSS="Perceptual"
LAYERS_VGG16_LOSS_PERCEPTUAL="[block1_conv1,block2_conv1,block3_conv1]"
WEIGHTS_VGG16_LOSS_PERCEPTUAL="[0.65,0.30,0.05]"
PROP_REDSIZE_VGG16_LOSS_PERCEPTUAL="0.0"
# Use approx. the ratio between the CONVERGED TRAIN LOSSES from an experiment running ONLY the FIRST over an experiment running ONLY the SECOND loss
#WEIGHT_COMBINED_LOSS="0.003"   # for DSSIM + L1
WEIGHT_COMBINED_LOSS="0.00006"   # for DSSIM + L2
#WEIGHT_COMBINED_LOSS="10.0"   # for Perceptual + L1
#WEIGHT_COMBINED_LOSS="0.2"   # for Perceptual + L2
LIST_TYPE_METRICS="[DSSIM]"
TYPE_OPTIMIZER="Adam"
LEARN_RATE="1.0e-04"
BATCH_SIZE="1"
NUM_EPOCHS="1000"
IS_MASK_REGION_INTEREST="False"
USE_SLIDING_WINDOW_IMAGES="False"
USE_TRANSFORM_RIGID_IMAGES="True"
IS_LOAD_AUGMENTED_IMAGES_PER_LABEL="False"
NUM_AUGMENTED_IMAGES_PER_LABEL="28"
IS_TRAIN_NETWORK_2D="False"
#
# ---------- SETTINGS ----------

python "${WORKDIR}/Code/scripts_experiments/train_model.py" \
	--basedir=${WORKDIR} \
	--modelsdir=${MODELS_DIR} \
	--training_datadir=${TRAINDATA_DIR} \
	--validation_datadir=${VALIDDATA_DIR} \
	--size_in_images=${SIZE_IMAGES} \
	--type_network=${TYPE_NETWORK} \
	--net_num_featmaps=${NET_NUM_FEATMAPS} \
	--type_loss=${TYPE_LOSS} \
	--weight_combined_loss=${WEIGHT_COMBINED_LOSS} \
	--list_type_metrics=${LIST_TYPE_METRICS} \
	--type_optimizer=${TYPE_OPTIMIZER} \
	--learn_rate=${LEARN_RATE} \
	--batch_size=${BATCH_SIZE} \
	--num_epochs=${NUM_EPOCHS} \
	--is_mask_region_interest=${IS_MASK_REGION_INTEREST} \
	--use_sliding_window_images=${USE_SLIDING_WINDOW_IMAGES} \
	--use_transform_rigid_images=${USE_TRANSFORM_RIGID_IMAGES} \
	--is_load_augmented_images_per_label=${IS_LOAD_AUGMENTED_IMAGES_PER_LABEL} \
	--num_augmented_images_per_label=${NUM_AUGMENTED_IMAGES_PER_LABEL} \
	--is_train_network_2D=${IS_TRAIN_NETWORK_2D} \
	--layers_vgg16_loss_perceptual=${LAYERS_VGG16_LOSS_PERCEPTUAL} \
	--weights_vgg16_loss_perceptual=${WEIGHTS_VGG16_LOSS_PERCEPTUAL} \
	--prop_redsize_vgg16_loss_perceptual=${PROP_REDSIZE_VGG16_LOSS_PERCEPTUAL} #\
	#--is_restart_model="True" \
	#--in_config_file="${MODELS_DIR}/configparams.txt" \
	#--is_restart_weights_diffmodel="True"
