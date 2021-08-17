#!/bin/bash

basedata="./GRASE2CUBE_Processed/"

ln -s $basedata "./BaseData"

# 1: distribute data
modelbasedir="./Models_TestGRASE2CUBE/"

list_files_order_train=$(find "${modelbasedir}/CVfoldsInfo/" -name "train[0-9].txt")

for infile_order_train in $list_files_order_train
do
    ifold=$(echo $(basename $infile_order_train) | egrep -o "[[:digit:]]{1}")
    
    python3 "./Code/src/scripts_experiments/distribute_data.py" \
	    --basedir=. \
	    --type_data="testing" \
	    --type_distribute="orderfile" \
	    --infile_order_train=${infile_order_train}

    rm -r "./TrainingData/" "./ValidationData/" && mv "./TestingData/" "./TestingData_CV${ifold}/"
done

# 2: run predictions
modelname="model_converged.hdf5"
outputbasedir="./Test_Predictions_GRASE2CUBE/"

mkdir -p "${outputbasedir}/Posteriors/"

list_modeldirs=$(find "${modelbasedir}" -name "Models_GRASE2CUBE_CV[0-9]")

for modeldir in $list_modeldirs
do
    ifold=$(echo $(basename $modeldir) | sed 's/GRASE2CUBE//' | egrep -o "[[:digit:]]{1}")

    modeltest="${modeldir}/${modelname}"
    outputdir="${outputbasedir}/Predictions_CV${ifold}"

    python3 "./Code/src/scripts_experiments/predict_model.py" $modeltest \
	    --basedir="." \
	    --in_config_file="${modeldir}/configparams.txt" \
	    --testing_datadir="./TestingData_CV${ifold}/" \
	    --name_output_predictions_relpath="${outputdir}/PosteriorsWorkData/" \
	    --name_output_reference_keys_file="${outputdir}/referenceKeys_posteriors.npy"

    python3 "./Code/src/scripts_evalresults/postprocess_predictions.py" \
	    --basedir="." \
	    --name_input_predictions_relpath="${outputdir}/PosteriorsWorkData/" \
	    --name_input_reference_keys_file="${outputdir}/referenceKeys_posteriors.npy" \
	    --name_output_posteriors_relpath="${outputdir}/Posteriors/"

    mv "${outputdir}/Posteriors/"* "${outputbasedir}/Posteriors/"
    rm -r "${outputdir}"
done

# 3: clean-up
rm -r "./TestingData_CV"[0-9]
rm "./BaseData"
