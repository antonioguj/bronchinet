#!/bin/bash

# Load the modules
module purge
module load Python/3.7.4-GCCcore-8.3.0

source /tmp/${SLURM_JOB_USER}.${SLURM_JOB_ID}/prolog.env

HOME="/trinity/home/agarcia/"
WORKDIR="${HOME}/Results/MRIenhanceTests/"

export PYTHONPATH="${WORKDIR}/Code/:${PYTHONPATH}"

# Load python virtual environment
source "${HOME}/Pyvenv-v.3.7.4/bin/activate"

# SETTINGS
IMAGESDATA_DIR="${WORKDIR}/BaseData/ImagesWorkData/"
LABELSDATA_DIR="${WORKDIR}/BaseData/LabelsWorkData_Masked/"
TRAINDATA_OUTDIR="${WORKDIR}/TrainingData_Masked/"
VALIDDATA_OUTDIR="${WORKDIR}/ValidationData_Masked/"
TESTDATA_OUTDIR="${WORKDIR}/TestingData_Masked/"
TYPE_DATA="training"
TYPE_DISTRIBUTE="original"
DIST_PROPDATA_TRAIN_VALID_TEST="(0.65,0.15,0.20)"
INFILE_ORDER_TRAIN="${WORKDIR}/train.txt"
IS_PREPARE_MANY_IMAGES_PER_LABEL="False"
# --------

python3 "${WORKDIR}/Code/scripts_experiments/distribute_data.py" \
    --basedir=${WORKDIR} \
    --type_data=${TYPE_DATA} \
    --type_distribute=${TYPE_DISTRIBUTE} \
    --dist_propdata_train_valid_test=${DIST_PROPDATA_TRAIN_VALID_TEST} \
    --infile_order_train=${INFILE_ORDER_TRAIN} \
    --is_prepare_many_images_per_label=${IS_PREPARE_MANY_IMAGES_PER_LABEL}
    --name_input_images_relpath=${IMAGESDATA_DIR} \
    --name_input_labels_relpath=${LABELSDATA_DIR} \
    --name_training_data_relpath=${TRAINDATA_OUTDIR} \
    --name_validation_data_relpath=${VALIDDATA_OUTDIR} \
    --name_testing_data_relpath=${TESTDATA_OUTDIR}
