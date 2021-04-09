#!/bin/bash

basedata="./EXACT_Testset_Testing/"

ln -s $basedata "./BaseData"

# 1: distribute data
modeldir="./Models_TestEXACT/"

python3 "./Code/scripts_experiments/distribute_data.py" \
	--basedir=. \
	--type_data="testing" \
	--type_distribute="original" \
	--propdata_train_valid_test="(0.0,0.0,1.0)"

rm -r "./TrainingData/" "./ValidationData/"

# 2: run predictions
modeltest="${modeldir}/model_converged.pt"
outputdir="./Test_Predictions_EXACT/"

python3 "./Code/scripts_launch/launch_predictions_full.py" $modeltest $outputdir \
	--post_threshold_value="0.1" \
	--is_connected_masks="True" \
	--in_connregions_dim="1" \
	--list_type_metrics_result="[]" \
	--is_backward_compat="True"

# 3: clean-up
rm -r "./TestingData/"
rm "./BaseData"
