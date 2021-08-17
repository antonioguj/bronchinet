#!/bin/bash

codedir="/home/antonio/Codes/bronchinet/"

basedata="./EXACT_Testset_Testing/"

ln -s $basedata "./BaseData"


# 1: DISTRIBUTE DATA
modeldir="./Models_TestEXACT/"

python3 "${codedir}/src/scripts_experiments/distribute_data.py" \
	--basedir=. \
	--type_data="testing" \
	--type_distribute="original" \
	--propdata_train_valid_test="(0.0,0.0,1.0)"

rm -r "./TrainingData/" "./ValidationData/"


# 2: RUN PREDICTIONS
modeltest="${modeldir}/model_converged.pt"
outputdir="./Test_Predictions_EXACT/"

python3 "${codedir}/scripts/launch_predictions_full.py" $modeltest $outputdir \
	--post_threshold_value="0.1" \
	--is_connected_masks="True" \
	--in_connregions_dim="1" \
	--list_type_metrics_result="[]" \
	--is_backward_compat="True"


# 3: CLEAN-UP
rm -r "./TestingData/"
rm "./BaseData"
