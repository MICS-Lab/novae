#!/bin/bash

BUCKET_NAME="vz-ultra-showcase"
OUTPUT_DIR="./merscope"

mkdir -p $OUTPUT_DIR

for BUCKET_FILE in $(gsutil ls -d gs://$BUCKET_NAME/*/region_*/*.h5ad); do
    DATASET_NAME=$(basename $BUCKET_FILE)

    if [ -f $OUTPUT_DIR/$DATASET_NAME ]; then
        echo "File $DATASET_NAME already exists in $OUTPUT_DIR"
    else
        echo "Copying $BUCKET_FILE to $OUTPUT_DIR"
        gsutil cp "$BUCKET_FILE" $OUTPUT_DIR
        echo "Copied successfully"
    fi
done
