#!/bin/bash

if [ "$1" == "" ] || [ "$2" == "" ] || [ "$3" == "" ]
then
    echo "ERROR: Usage: \"$0\" \"INPUT_DATA_DIR\" \"OUTPUT_DIR\" \"TYPE_BACKEND\" (= [--torch, --keras])"
    exit 1
fi


input_data_dir=$1
output_dir=$2
type_backend=$3
workdir=$PWD

if [ "$type_backend" == "--torch" ]
then
    in_rel_model_file="model_trained_torch.pt"
    # set "type_backend == torch" in code
    #sed -i "s/TYPE_DNNLIB_USED = 'Keras'/TYPE_DNNLIB_USED = 'Pytorch'/" ~/Codes/bronchinet/src/common/constant.py
    is_old_trained_model="True"
elif [ "$type_backend" == "--keras" ]
then
    in_rel_model_file="model_trained_keras.hdf5"
    # set "type_backend == keras" in code
    #sed -i "s/TYPE_DNNLIB_USED = 'Pytorch'/TYPE_DNNLIB_USED = 'Keras'/" ~/Codes/bronchinet/src/common/constant.py
    is_old_trained_model="False"
else
    echo "ERROR: input \"TYPE_BACKEND\" not either \"--torch\" or \"--keras\""
    exit 1
fi

if [ ! -d "${workdir}/Code" ] || [ ! -L "${workdir}/Code" ]
then
    echo "ERROR: simlink "./Code" to our code sources does not exist"
    exit 1
fi

if [ ! -d "${workdir}/BaseData" ] || [ ! -L "${workdir}/BaseData" ]
then
    echo "WARNING: simlink "./BaseData" to your input data does not exist. Create simlink to input \"INPUT_DATA_DIR\""
    ln -s $input_data_dir "${workdir}/BaseData"
fi


# User-defined settings
val_thres_probs="0.1"  		# value to threshold probability maps
is_conn_binmasks="True"         # compute first connected component from airway segmentation ?
is_mask_lungs="True"            # mask output probability maps to lung mask, to remove noise ?
is_coarse_airways="True"        # include mask of coarse airways (trachea & main bronchi) ?
# ----------


# Paths to input data needed / output results
in_images_dir="${workdir}/BaseData/Images/"
in_lungmasks_dir="${workdir}/BaseData/Lungs/"
in_coarseairways_dir="${workdir}/BaseData/CoarseAirways/"
in_model_file="${workdir}/models/${in_rel_model_file}"
in_config_file="${workdir}/models/configparams.txt"

if [ ! -d "$output_dir" ] 
then
    mkdir -p $output_dir
fi


# 0. Arrange input images in data format expected by our code (avoid running script "prepareData.py")
in_testdata_dir="${workdir}/TestingData/"
in_referkeys_file="${workdir}/referenceKeys.csv"

mkdir -p $in_testdata_dir
touch $in_referkeys_file

echo "Create links to input images in directory: ./TestingData..."

list_image_files=$(find $in_images_dir -type f | sort -n)

count="1"
for in_file in $list_image_files
do
    in_link_file=$(printf "${in_testdata_dir}/images_proc-%02i.nii.gz" "$count")

    # create link to input image
    echo "$in_file --> ${in_link_file}"
    ln -s "$in_file" $in_link_file

    # append new reference key
    echo "$(basename ${in_link_file%.nii.gz}),$(basename $in_file)" >> $in_referkeys_file

    count=$((count+1))
done

echo ""
# ----------


# 1. Compute predictions of probability maps from the trained model
python3 "${workdir}/Code/src/scripts_experiments/predict_model.py" ${in_model_file} \
        --basedir=${workdir} \
        --in_config_file=${in_config_file} \
        --testing_datadir=${in_testdata_dir} \
	--name_output_predictions_relpath="${output_dir}/PosteriorsWorkData/" \
	--name_input_reference_keys_file=${in_referkeys_file} \
        --name_output_reference_keys_file="${output_dir}/referenceKeys_posteriors.npy" \
        --is_backward_compat="${is_old_trained_model}"
echo ""


# 2. Compute full-size probability maps from the cropped prob maps
python3 "${workdir}/Code/src/scripts_evalresults/postprocess_predictions.py" \
        --basedir=${workdir} \
	--is_mask_region_interest=${is_mask_lungs} \
        --is_crop_images="False" \
	--name_input_predictions_relpath="${output_dir}/PosteriorsWorkData/" \
        --name_output_posteriors_relpath="${output_dir}/Posteriors/" \
	--name_input_reference_files_relpath=${in_images_dir} \
	--name_input_reference_keys_file="${output_dir}/referenceKeys_posteriors.npy" \
	--name_input_roimasks_relpath=${in_lungmasks_dir}
echo ""


# 3. Compute binary segmentation from the full-size probability maps
python3 "${workdir}/Code/src/scripts_evalresults/process_predicted_airway_tree.py" \
	--basedir=${workdir} \
	--post_threshold_value=${val_thres_probs} \
	--is_attach_coarse_airways=${is_coarse_airways} \
	--name_input_posteriors_relpath="${output_dir}/Posteriors/" \
	--name_output_binary_masks_relpath="${output_dir}/BinaryMasks/" \
	--name_input_reference_keys_file=${in_referkeys_file} \
	--name_input_coarse_airways_relpath=${in_coarseairways_dir}
echo ""


# 4. Compute first connected component from the binary segmentation
if [ "$is_conn_binmasks" == "True" ]
then
    python3 "${workdir}/Code/src/scripts_util/apply_operation_images.py" "${output_dir}/BinaryMasks/" "${output_dir}/BinaryMasks_Connected/" \
	    --type="firstconreg" \
	    --in_conreg_dim="1"
    echo ""

    rm -r "${output_dir}/BinaryMasks/" && mv "${output_dir}/BinaryMasks_Connected/" "${output_dir}/BinaryMasks/"
fi


# 5. Clean-up tempo data
rm -r $in_testdata_dir
rm $in_referkeys_file
rm -r "${output_dir}/PosteriorsWorkData/"
rm "${output_dir}/referenceKeys_posteriors.npy" "${output_dir}/referenceKeys_posteriors.csv"
