
# export PYTHONPATH="${WORKDIR}/Code/:${PYTHONPATH}"

# PUT HERE THE PATHS TO YOUR WORKDIR
HOME="/home/antonio/"		
WORKDIR="${HOME}/Results/AirwaySegmentation_EXACT/"

# COPY DIR 'models/' FROM THE REPO TO THE WORKDIR

MODEL_TEST="${WORKDIR}/models/model_evalEXACT.pt"
CONFIG_FILE="${WORKDIR}/models/configparams.txt"

TESTDATA_DIR="${WORKDIR}/TestingData/"
OUTPUT_DIR="${WORKDIR}/Predictions/"


mkdir -p ${OUTPUT_DIR}

python3 "${WORKDIR}/Code/scripts_experiments/predict_model.py" ${MODEL_TEST} \
        --basedir=${WORKDIR} \
        --in_config_file=${CONFIG_FILE} \
        --testing_datadir=${TESTDATA_DIR} \
	--name_output_predictions_relpath="${OUTPUT_DIR}/PosteriorsWorkData/" \
        --name_output_reference_keys_file="${OUTPUT_DIR}/referenceKeys_posteriors.npy" \
        --is_backward_compat="True"

python3 "${WORKDIR}/Code/scripts_evalresults/postprocess_predictions.py" \
        --basedir=${WORKDIR} \
	--is_mask_region_interest="True" \
        --is_crop_images="True" \
	--name_input_predictions_relpath="${OUTPUT_DIR}/PosteriorsWorkData/" \
        --name_output_posteriors_relpath="${OUTPUT_DIR}/Posteriors/" \
	--name_input_reference_keys_file="${OUTPUT_DIR}/referenceKeys_posteriors.npy" \

python3 "${WORKDIR}/Code/scripts_evalresults/process_predicted_airway_tree.py" \
	--basedir=${WORKDIR} \
	--post_threshold_values="0.1" \
	--is_attach_coarse_airways="True" \
	--name_input_posteriors_relpath="${OUTPUT_DIR}/Posteriors/" \
	--name_output_binary_masks_relpath="${OUTPUT_DIR}/BinaryMasks/"

python3 "${WORKDIR}/Code/scripts_util/apply_operation_images.py" "${OUTPUT_DIR}/BinaryMasks/" "${OUTPUT_DIR}/BinaryMasks_Connected/" \
	--type="firstconreg" \
	--in_conreg_dim="1"

rm -r "${OUTPUT_DIR}/BinaryMasks/" && mv "${OUTPUT_DIR}/BinaryMasks_Connected/" "${OUTPUT_DIR}/BinaryMasks/"

rm -r "${OUTPUT_DIR}/PosteriorsWorkData/"
rm "${OUTPUT_DIR}/referenceKeys_posteriors.npy" "${OUTPUT_DIR}/referenceKeys_posteriors.csv"
