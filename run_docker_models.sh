#!/bin/bash

if [ "$1" == "" ] || [ "$2" == "" ] || [ "$3" == "" ]
then
    echo "ERROR: Usage: \"$0\" \"INPUT_DATA_DIR\" \"OUTPUT_DIR\" \"NAME_IMAGE_VERSION\" (= e.g. bronchinet:latest)"
    exit 1
fi

INPUT_DATA_DIR=$1
OUTPUT_DIR=$2
NAME_IMAGE_VERSION=$3

if [ ! -d "$OUTPUT_DIR" ]
then
    mkdir -p $OUTPUT_DIR
fi

CALL="sudo docker run --gpus all --rm -it -v ${INPUT_DATA_DIR}:/workdir/input_data/ -v ${OUTPUT_DIR}:/workdir/results/ ${NAME_IMAGE_VERSION}"
echo -e "$CALL"
eval "$CALL"
