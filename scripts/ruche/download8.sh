#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu_med

module purge
cd /gpfs/workdir/blampeyq/novae/data


OUTPUT_DIR="./xenium"

mkdir -p $OUTPUT_DIR

# Last dataset release date: 2024-05-28
ZIP_REMOTE_PATHS=(\
    "https://cf.10xgenomics.com/samples/xenium/1.0.2/Xenium_V1_FFPE_Human_Breast_ILC/Xenium_V1_FFPE_Human_Breast_ILC_outs.zip"\
    "https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.0.2/Xenium_V1_FFPE_Human_Breast_IDC/Xenium_V1_FFPE_Human_Breast_IDC_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/1.0.2/Xenium_V1_FF_Mouse_Brain_MultiSection_1/Xenium_V1_FF_Mouse_Brain_MultiSection_1_outs.zip"\
    # "https://cf.10xgenomics.com/samples/xenium/1.0.2/Xenium_V1_FF_Mouse_Brain_MultiSection_2/Xenium_V1_FF_Mouse_Brain_MultiSection_2_outs.zip"\
    # "https://cf.10xgenomics.com/samples/xenium/1.0.2/Xenium_V1_FF_Mouse_Brain_MultiSection_3/Xenium_V1_FF_Mouse_Brain_MultiSection_3_outs.zip"\
    # "https://cf.10xgenomics.com/samples/xenium/1.9.0/Xenium_V1_mFemur_EDTA_3daydecal_section/Xenium_V1_mFemur_EDTA_3daydecal_section_outs.zip"\
    # "https://cf.10xgenomics.com/samples/xenium/1.9.0/Xenium_V1_mFemur_EDTA_PFA_3daydecal_section/Xenium_V1_mFemur_EDTA_PFA_3daydecal_section_outs.zip"\
)

for ZIP_REMOTE_PATH in "${ZIP_REMOTE_PATHS[@]}"
do
    DATASET_NAME=$(basename $ZIP_REMOTE_PATH)
    OUTPUT_DATASET_ZIP=$OUTPUT_DIR/${DATASET_NAME}
    OUTPUT_DATASET_DIR="${OUTPUT_DATASET_ZIP%.zip}"

    if [ -d $OUTPUT_DATASET_DIR ]; then
        echo "Directory $OUTPUT_DATASET_DIR already exists"
    else
        if [ -f $OUTPUT_DATASET_ZIP ]; then
            echo "File $OUTPUT_DATASET_ZIP already exists"
        else
            echo "Downloading $ZIP_REMOTE_PATH to $OUTPUT_DATASET_ZIP"
            curl $ZIP_REMOTE_PATH -o $OUTPUT_DATASET_ZIP
            echo "Successfully downloaded"
        fi
        echo "Unzipping in $OUTPUT_DATASET_DIR"
        mkdir -p $OUTPUT_DATASET_DIR
        /gpfs/workdir/blampeyq/unzip -j $OUTPUT_DATASET_ZIP cell_feature_matrix.h5 cells.parquet -d $OUTPUT_DATASET_DIR
        echo "Successfully unzipped files"
        rm $OUTPUT_DATASET_ZIP
    fi
done
