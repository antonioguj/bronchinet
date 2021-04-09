#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH -p hm
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
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
WORKDIR="${HOME}/Results/MRIenhanceTests/"

export PYTHONPATH="${WORKDIR}/Code/:${PYTHONPATH}"

# Load virtual environment
source "${HOME}/Pyvenv-v.3.7.4/bin/activate"

# SETTINGS
MODELS_DIR="${WORKDIR}/SavedModels/Models_DSSIM/"
MODEL_TEST="${MODELS_DIR}/model_last.hdf5"	# or in general model_e<jobnum>.hdf5
TESTDATA_DIR="${WORKDIR}/TestingData_Fixed/"
OUTPUT_DIR="${WORKDIR}/Predictions_DSSIM/"
OUT_METRICS_FILE="${OUTPUT_DIR}/res_metrics.csv"
IS_MASK_REGION_INTEREST="False"
IS_BACKWARD_COMPAT="False"	# only set "True" if it fails with "False"
IS_TEST_NETWORK_2D="False"
# --------

mkdir -p $OUTPUT_DIR

python3 "${WORKDIR}/Code/scripts_experiments/predict_model.py" ${MODEL_TEST} \
	--basedir=${WORKDIR} \
	--in_config_file="${MODELS_DIR}/configparams.txt" \
	--testing_datadir=${TESTDATA_DIR} \
	--name_output_predictions_relpath="${OUTPUT_DIR}/PosteriorsWorkData/" \
	--name_output_reference_keys_file="${OUTPUT_DIR}/referenceKeys_posteriors.npy" \
	--is_backward_compat=${IS_BACKWARD_COMPAT} \
	--is_test_network_2D=${IS_TEST_NETWORK_2D}

python3 "${WORKDIR}/Code/scripts_evalresults/postprocess_predictions.py" \
	--basedir=${WORKDIR} \
	--name_input_predictions_relpath="${OUTPUT_DIR}/PosteriorsWorkData/" \
	--name_input_reference_keys_file="${OUTPUT_DIR}/referenceKeys_posteriors.npy" \
	--name_output_posteriors_relpath="${OUTPUT_DIR}/Posteriors/" \
	--is_mask_region_interest=${IS_MASK_REGION_INTEREST}

python3 "${WORKDIR}/Code/scripts_evalresults/compute_result_metrics.py" "${OUTPUT_DIR}/Posteriors/" \
	--basedir=${WORKDIR} \
	--output_file=${OUT_METRICS_FILE}

rm -r "${OUTPUT_DIR}/PosteriorsWorkData/"
rm "${OUTPUT_DIR}/referenceKeys_posteriors.npy" "${OUTPUT_DIR}/referenceKeys_posteriors.csv"
