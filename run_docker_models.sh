#!/bin/bash

if [ "$1" == "" ] || [ "$2" == "" ]
then
    echo "ERROR: Usage: \"$0\" \"INPUT_DATA_DIR\" \"OUTPUT_DIR\""
    exit 1
fi

INPUT_DATA_DIR=$1
OUTPUT_DIR=$2

CALL="docker run --user ${UID}:${GID} --gpus all -ti -v ${INPUT_DATA_DIR}:/workdir/input_data/ -v ${OUTPUT_DIR}:/workdir/results/"
echo -e "$CALL"
eval "$CALL"
