#!/bin/bash

BUCKET_NAME="vz-ffpe-showcase"
OUTPUT_DIR="./merscope"

for BUCKET_DIR in $(gsutil ls -d gs://$BUCKET_NAME/*); do
    DATASET_NAME=$(basename $BUCKET_DIR)
    OUTPUT_DATASET_DIR=$OUTPUT_DIR/$DATASET_NAME
    
    mkdir -p $OUTPUT_DATASET_DIR
    
    for BUCKET_FILE in ${BUCKET_DIR}{cell_by_gene,cell_metadata}.csv; do
        FILE_NAME=$(basename $BUCKET_FILE)

        if [ -f $OUTPUT_DATASET_DIR/$FILE_NAME ]; then
            echo "File $FILE_NAME already exists in $OUTPUT_DATASET_DIR"
        else
            echo "Copying $BUCKET_FILE to $OUTPUT_DATASET_DIR"
            gsutil cp "$BUCKET_FILE" $OUTPUT_DATASET_DIR
            echo "Copied successfully"
        fi
    done
done
