
HOME="/home/antonio/"
WORKDIR="${HOME}/Results/AirwaySegmentation_EXACT/"

# export PYTHONPATH="${WORKDIR}/Code/:${PYTHONPATH}"

MODEL_TEST="${WORKDIR_DIR}/model_evalEXACT.pt"
CONFIG_FILE="${WORKDIR_DIR}/configparams.txt"
OUTPUT_DIR="${WORKDIR}/Predictions/"


python3 "${WORKDIR}/Code/scripts_experiments/predict_model.py" ${MODEL_TEST} \
        --basedir=${WORKDIR} \
        --in_config_file=${CONFIG_FILE} \
        --testing_datadir="TestingData/" \
        --is_backward_compat="True"

python3 "${WORKDIR}/Code/scripts_evalresults/postprocess_predictions.py" \
        --basedir=${WORKDIR}

python3 "${WORKDIR}/Code/scripts_evalresults/process_predicted_airway_tree.py" \
	--basedir=${WORKDIR} \
	--post_threshold_values="0.1" \
	--is_attach_coarse_airways="True"

python3 "${WORKDIR}/Code/scripts_util/apply_operation_images.py" "${OUTPUT_DIR}/BinaryMasks/" "${OUTPUT_DIR}/BinaryMasks_Connected/" \
	--type="firstconreg" \
	--in_conreg_dim="1"

rm -r "${OUTPUT_DIR}/BinaryMasks/" && mv "${OUTPUT_DIR}/BinaryMasks_Connected/" "${OUTPUT_DIR}/BinaryMasks/"
