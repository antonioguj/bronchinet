#!/bin/bash

# Load the modules
module purge
module load Python/3.7.4-GCCcore-8.3.0

source /tmp/${SLURM_JOB_USER}.${SLURM_JOB_ID}/prolog.env


HOME="/trinity/home/agarcia/"
WORKDIR="${HOME}/Results/AirwaySegmentation_IBEST/"

export PYTHONPATH="${PYTHONPATH}:${WORKDIR}/Code/"

# Load python virtual environment
source "${HOME}/Pyvenv-v.3.7.4/bin/activate"


# ---------- SETTINGS ----------
#
IMAGESDATA_DIR="${WORKDIR}/BaseData/ImagesWorkData/"
LABELSDATA_DIR="${WORKDIR}/BaseData/LabelsWorkData/"
TRAINDATA_OUTDIR="${WORKDIR}/TrainingData/"
VALIDDATA_OUTDIR="${WORKDIR}/ValidationData/"
TESTDATA_OUTDIR="${WORKDIR}/TestingData/"
TYPE_DISTRIBUTE="orderfile"
DIST_PROPDATA_TRAIN_VALID_TEST="(0.65,0.15,0.20)"
#
# ---------- SETTINGS ----------

python "${WORKDIR}/Code/scripts_experiments/distribute_data.py" \
	--basedir=${WORKDIR} \
	--name_input_images_relpath=${IMAGESDATA_DIR} \
	--name_input_labels_relpath=${LABELSDATA_DIR} \
	--name_training_data_relpath=${TRAINDATA_OUTDIR} \
	--name_validation_data_relpath=${VALIDDATA_OUTDIR} \
	--name_testing_data_relpath=${TESTDATA_OUTDIR} \
	--type_data="training" \
	--type_distribute=${TYPE_DISTRIBUTE} \
	--dist_propdata_train_valid_test=${DIST_PROPDATA_TRAIN_VALID_TEST}
	#--infile_train_order="${WORKDIR}/train.txt"
