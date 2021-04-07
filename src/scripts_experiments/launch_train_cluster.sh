#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=14G
#SBATCH -p hm
#SBATCH --gres=gpu:1
#SBATCH -t 10-00:00:00
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
WORKDIR="${HOME}/Results/AirwaySegmentation_DLCST-LUVAR/"

export PYTHONPATH="${WORKDIR}/Code/:${PYTHONPATH}"

# Load virtual environment
source "${HOME}/Pyvenv-v.3.7.4/bin/activate"


MODELS_DIR="${WORKDIR}/Models/"
TRAINDATA_DIR="${WORKDIR}/TrainingData/"
VALIDDATA_DIR="${WORKDIR}/ValidationData/"

python "${WORKDIR}/Code/scripts_experiments/train_model.py" \
	--basedir=${WORKDIR} \
	--modelsdir=${MODELS_DIR} \
	--training_datadir=${TRAINDATA_DIR} \
	--validation_datadir=${VALIDDATA_DIR} #\
	#--is_restart_model="True" \
	#--in_config_file="${MODELS_DIR}/configparams.txt"
