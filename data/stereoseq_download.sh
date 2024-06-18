H5AD_REMOTE_PATHS=(\
    "https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000062/stomics/FP200000498TL_D2_stereoseq.h5ad"\
    "https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000062/stomics/FP200000498TL_E4_stereoseq.h5ad"\
    "https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000062/stomics/FP200000498TL_E5_stereoseq.h5ad"\
)

OUTPUT_DIR="stereoseq"
mkdir -p $OUTPUT_DIR

for H5AD_REMOTE_PATH in "${H5AD_REMOTE_PATHS[@]}"
do
    DATASET_NAME=$(basename $H5AD_REMOTE_PATH)
    OUTPUT_DATASET_DIR=${OUTPUT_DIR}/${DATASET_NAME%.h5ad}
    OUTPUT_DATASET=$OUTPUT_DATASET_DIR/${DATASET_NAME}

    mkdir -p $OUTPUT_DATASET_DIR

    if [ -f $OUTPUT_DATASET ]; then
        echo "File $OUTPUT_DATASET_DIR already exists"
    else
        echo "Downloading $H5AD_REMOTE_PATH to $OUTPUT_DATASET"
        # curl $H5AD_REMOTE_PATH -o $OUTPUT_DATASET
        echo "Successfully downloaded"
    fi
done
