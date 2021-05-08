#!/bin/bash

# SETTINGS (TO CHANGE BY THE USER)
# --------
#
# PATHS TO THE REPO AND YOUR WORK DIRECTORY
CODEDIR="/home/antonio/Codes/bronchinet/"
WORKDIR="/home/antonio/Results/Bronchinet_Test/"

VAL_THRES_PROBS="0.1"		# VALUE TO THRESHOLD PROB MAPS
IS_CONN_BINMASKS="True"		# COMPUTE CONNECTED COMPONENT OF BINARY MASK ?
IS_MASK_LUNGS="True"		# MASK OUTPUT PROB MAPS TO LUNG MASK TO REMOVE NOISE ?
IS_COARSE_AIRWAYS="True"	# INCLUDE MASK OF COARSE AIRWAYS (TRACHEA & MAIN BRONCHI) ?

# PATH TO THE INPUT TEST IMAGES / OUTPUT PREDICTIONS
INPUT_IMAGES_DIR="${WORKDIR}/InputImages/"
OUTPUT_DIR="${WORKDIR}/Output_Predictions/"

# PATH TO THE LUNG MASKS AND COARSE AIRWAYS (IN CASE IT'S USED) FOR ALL TEST DATA
INPUT_LUNGMASKS_DIR="${WORKDIR}/Lungs/"
INPUT_COARSEAIRWAYS_DIR="${WORKDIR}/CoarseAirways/"
# --------


export PYTHONPATH="${CODEDIR}/src/:${PYTHONPATH}"

MODEL_TEST="${CODEDIR}/models/model_evalEXACT.pt"
CONFIG_FILE="${CODEDIR}/models/configparams.txt"

mkdir -p $OUTPUT_DIR


# NEEDED WHEN TESTING DIRECTLY FROM INPUT CT SCANS (WITHOUT RUNNING SCRIPT "<>/prepareData.py")
#
BASEDATADIR="${WORKDIR}/BaseData/"		# DUMMY, NOT USED, BUT CODE COMPLAINS OTHERWISE
INPUT_TESTDATA_DIR="${WORKDIR}/TestingData/"
INPUT_REFERKEYS_FILE="${WORKDIR}/referenceKeys.csv"

mkdir -p $BASEDATADIR
mkdir -p $INPUT_TESTDATA_DIR
touch $INPUT_REFERKEYS_FILE

echo "Create links to input images in directory: ./TestingData/..."

LIST_INPUT_FILES=$(find $INPUT_IMAGES_DIR -type f | sort -n)

COUNT="1"
for IN_FILE in $LIST_INPUT_FILES
do
    OUT_LINK_FILE=$(printf "${INPUT_TESTDATA_DIR}/images_proc-%02i.nii.gz" "$COUNT")

    # CREATE LINK TO INPUT FILE
    echo "$IN_FILE --> ${OUT_LINK_FILE}"
    ln -s $IN_FILE $OUT_LINK_FILE

    # APPEND NEW REFER_KEY IN FILE
    echo "$(basename ${OUT_LINK_FILE%.nii.gz}),$(basename $IN_FILE)" >> $INPUT_REFERKEYS_FILE

    COUNT=$((COUNT+1))
done

echo ""
# ----------


python3 "${CODEDIR}/src/scripts_experiments/predict_model.py" ${MODEL_TEST} \
        --basedir=${WORKDIR} \
        --in_config_file=${CONFIG_FILE} \
        --testing_datadir=${INPUT_TESTDATA_DIR} \
	--name_output_predictions_relpath="${OUTPUT_DIR}/PosteriorsWorkData/" \
	--name_input_reference_keys_file=${INPUT_REFERKEYS_FILE} \
        --name_output_reference_keys_file="${OUTPUT_DIR}/referenceKeys_posteriors.npy" \
        --is_backward_compat="True"
echo ""


python3 "${CODEDIR}/src/scripts_evalresults/postprocess_predictions.py" \
        --basedir=${WORKDIR} \
	--is_mask_region_interest=${IS_MASK_LUNGS} \
        --is_crop_images="False" \
	--name_input_predictions_relpath="${OUTPUT_DIR}/PosteriorsWorkData/" \
        --name_output_posteriors_relpath="${OUTPUT_DIR}/Posteriors/" \
	--name_input_reference_files_relpath=${INPUT_IMAGES_DIR} \
	--name_input_reference_keys_file="${OUTPUT_DIR}/referenceKeys_posteriors.npy" \
	--name_input_roimasks_relpath=${INPUT_LUNGMASKS_DIR}
echo ""


python3 "${CODEDIR}/src/scripts_evalresults/process_predicted_airway_tree.py" \
	--basedir=${WORKDIR} \
	--post_threshold_value=${VAL_THRES_PROBS} \
	--is_attach_coarse_airways=${IS_COARSE_AIRWAYS} \
	--name_input_posteriors_relpath="${OUTPUT_DIR}/Posteriors/" \
	--name_output_binary_masks_relpath="${OUTPUT_DIR}/BinaryMasks/" \
	--name_input_reference_keys_file=${INPUT_REFERKEYS_FILE} \
	--name_input_coarse_airways_relpath=${INPUT_COARSEAIRWAYS_DIR}
echo ""


if [ "$IS_CONN_BINMASKS" == "True" ]
then
    python3 "${CODEDIR}/src/scripts_util/apply_operation_images.py" "${OUTPUT_DIR}/BinaryMasks/" "${OUTPUT_DIR}/BinaryMasks_Connected/" \
	    --type="firstconreg" \
	    --in_conreg_dim="1"
    echo ""

    rm -r "${OUTPUT_DIR}/BinaryMasks/" && mv "${OUTPUT_DIR}/BinaryMasks_Connected/" "${OUTPUT_DIR}/BinaryMasks/"
fi


# CLEAN-UP TEMPO DATA
rm -r $BASEDATADIR
rm -r $INPUT_TESTDATA_DIR
rm $INPUT_REFERKEYS_FILE
rm -r "${OUTPUT_DIR}/PosteriorsWorkData/"
rm "${OUTPUT_DIR}/referenceKeys_posteriors.npy" "${OUTPUT_DIR}/referenceKeys_posteriors.csv"
