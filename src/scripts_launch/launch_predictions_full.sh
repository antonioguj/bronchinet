#!/bin/bash

workdir=$PWD

export PYTHONPATH="${workdir}/Code/:${PYTHONPATH}"

if [ "$1" == "" ] || [ "$2" == "" ]
then
    echo "ERROR: Usage: \"$0\" \"IN_MODEL_TEST\" \"OUTPUT_DIR\""
    exit 1
fi

in_model_test=$1
output_dir=$2
in_models_dir=$(dirname $in_model_test)

in_testdata_dir="${workdir}/TestingData/"
is_calc_metrics="False"
out_metrics_file="${output_dir}/res_metrics.csv"
is_mask_region_interest="False"
is_backward_compat="False"			# only set "True" if it fails with "False"
is_test_network_2D="False"

mkdir -p $output_dir

python3 "${workdir}/Code/scripts_experiments/predict_model.py" ${in_model_test} \
	--basedir=${workdir} \
	--in_config_file="${in_models_dir}/configparams.txt" \
	--testing_datadir=${in_testdata_dir} \
	--name_output_predictions_relpath="${output_dir}/PosteriorsWorkData/" \
	--name_output_reference_keys_file="${output_dir}/referenceKeys_posteriors.npy" \
	--is_backward_compat=${is_backward_compat} \
	--is_test_network_2D=${is_test_network_2D}

python3 "${workdir}/Code/scripts_evalresults/postprocess_predictions.py" \
	--basedir=${workdir} \
	--name_input_predictions_relpath="${output_dir}/PosteriorsWorkData/" \
	--name_input_reference_keys_file="${output_dir}/referenceKeys_posteriors.npy" \
	--name_output_posteriors_relpath="${output_dir}/Posteriors/" \
	--is_mask_region_interest=${is_mask_region_interest}

if [ "$is_calc_metrics" == "True" ]
then
    python3 "${workdir}/Code/scripts_evalresults/compute_result_metrics.py" "${output_dir}/Posteriors/" \
	    --basedir=${workdir} \
	    --output_file=${out_metrics_file}
fi

rm -r "${output_dir}/PosteriorsWorkData/"
rm "${output_dir}/referenceKeys_posteriors.npy" "${output_dir}/referenceKeys_posteriors.csv"
