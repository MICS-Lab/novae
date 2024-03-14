#!/bin/bash

OUTPUT_DIR="./xenium"
XENIUM_HTML_BASE_PATH=https://cf.10xgenomics.com/samples/xenium

mkdir -p $OUTPUT_DIR

ZIP_SUFFIXES=(\
    "1.3.0/Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon/Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs.zip"\
)

for ZIP_SUFFIX in "${ZIP_SUFFIXES[@]}"
do
    DATASET_DIR=$XENIUM_HTML_BASE_PATH/$ZIP_SUFFIX
    DATASET_NAME=$(basename $DATASET_DIR)
    OUTPUT_DATASET_ZIP=$OUTPUT_DIR/${DATASET_NAME}
    OUTPUT_DATASET_DIR="${OUTPUT_DATASET_ZIP%.zip}"

    if [ -f $OUTPUT_DATASET_ZIP ]; then
        echo "File $OUTPUT_DATASET_ZIP already exists"
    else
        echo "Donwloading $DATASET_DIR to $OUTPUT_DATASET_ZIP"
        curl $DATASET_DIR -o $OUTPUT_DATASET_ZIP
        echo "Successfully donwloaded"
    fi

    if [ -d $OUTPUT_DATASET_DIR ]; then
        echo "Directory $OUTPUT_DATASET_DIR already exists"
    else
        echo "Unzipping in $OUTPUT_DATASET_DIR"
        mkdir -p $OUTPUT_DATASET_DIR
        unzip -j $OUTPUT_DATASET_ZIP cell_feature_matrix.h5 cells.parquet -d $OUTPUT_DATASET_DIR
        echo "Successfully unzipped files"
    fi
done
