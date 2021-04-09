#!/bin/bash

basedata="./DLCST_Testing/"

ln -s $basedata "./BaseData"

# 1: distribute data
modelbasedir="./Models_TestDLCST/"

list_files_order_train=$(find "${modelbasedir}/CVfoldsInfo/" -name "train[0-9].txt")

for infile_order_train in $list_files_order_train
do
    ifold=$(echo $(basename $infile_order_train) | egrep -o "[[:digit:]]{1}")
    
    python3 "./Code/scripts_experiments/distribute_data.py" \
	    --basedir=. \
	    --type_data="testing" \
	    --type_distribute="orderfile" \
	    --infile_order_train=${infile_order_train}

    rm -r "./TrainingData/" "./ValidationData/" && mv "./TestingData/" "./TestingData_CV${ifold}/"
done

# 2: run predictions
modeltest="${modelbasedir}/Models_DLCST_CV1/model_converged.pt"
outputdir="./Test_Predictions_DLCST/"

python3 "./Code/scripts_launch/launch_predictions_full.py" $modeltest $outputdir \
	--testing_datadir="./TestingData_CV1/" \
	--is_preds_crossval="True" \
	--post_threshold_value="0.5" \
	--is_connected_masks="True" \
	--in_connregions_dim="3" \
	--is_backward_compat="True"

# 3: clean-up
rm -r "./TestingData_CV"[0-9]
rm "./BaseData"
