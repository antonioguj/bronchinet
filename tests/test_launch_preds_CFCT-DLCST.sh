#!/bin/bash

codedir="/home/antonio/Codes/bronchinet/"

basedata="./DLCST_Testing/"
#basedata="./LUVAR_Testing/"

ln -s $basedata "./BaseData"


# 1: DISTRIBUTE DATA
modelbasedir="./Models_TestDLCST/"
#modelbasedir="./Models_TestLUVAR/"

list_files_order_train=$(find "${modelbasedir}/CVfoldsInfo/" -name "train[0-9].txt")

for infile_order_train in $list_files_order_train
do
    ifold=$(echo $(basename $infile_order_train) | egrep -o "[[:digit:]]{1}")
    
    python3 "${codedir}/src/scripts_experiments/distribute_data.py" \
	    --basedir=. \
	    --type_data="testing" \
	    --type_distribute="orderfile" \
	    --infile_order_train=${infile_order_train}

    rm -r "./TrainingData/" "./ValidationData/" && mv "./TestingData/" "./TestingData_CV${ifold}/"
done


# 2: RUN PREDICTIONS
modeltest="${modelbasedir}/Models_DLCST_CV1/model_converged.pt"
outputdir="./Test_Predictions_DLCST/"
#modeltest="${modelbasedir}/Models_LUVAR_CV1/model_converged.pt"
#outputdir="./Test_Predictions_LUVAR/"

python3 "${codedir}/scripts/launch_predictions_full.py" $modeltest $outputdir \
	--testing_datadir="./TestingData_CV1/" \
	--is_preds_crossval="True" \
	--post_threshold_value="0.5" \
	--is_connected_masks="True" \
	--in_connregions_dim="3" \
	--is_backward_compat="True"


# 3: CLEAN-UP
rm -r "./TestingData_CV"[0-9]
rm "./BaseData"
