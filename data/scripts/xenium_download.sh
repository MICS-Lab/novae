#!/bin/bash

OUTPUT_DIR="./xenium"

mkdir -p $OUTPUT_DIR

# Last dataset release date: 2024-05-28
ZIP_REMOTE_PATHS=(\
    "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_Human_Colon_Cancer_P5_CRC_Add_on_FFPE/Xenium_V1_Human_Colon_Cancer_P5_CRC_Add_on_FFPE_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE/Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Pancreas_FFPE/Xenium_V1_human_Pancreas_FFPE_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_mouse_Colon_FF/Xenium_V1_mouse_Colon_FF_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/1.9.0/Xenium_V1_hBoneMarrow_acute_lymphoid_leukemia_section/Xenium_V1_hBoneMarrow_acute_lymphoid_leukemia_section_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/1.9.0/Xenium_V1_mFemur_formic_acid_24hrdecal_section/Xenium_V1_mFemur_formic_acid_24hrdecal_section_outs.zip"\
    "https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/2.0.0/Xenium_V1_Human_Brain_GBM_FFPE/Xenium_V1_Human_Brain_GBM_FFPE_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE/Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_Human_Colorectal_Cancer_Addon_FFPE/Xenium_V1_Human_Colorectal_Cancer_Addon_FFPE_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_Human_Lung_Cancer_Addon_FFPE/Xenium_V1_Human_Lung_Cancer_Addon_FFPE_outs.zip"\
    "https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/3.0.0/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_Human_Ovarian_Cancer_Addon_FFPE/Xenium_V1_Human_Ovarian_Cancer_Addon_FFPE_outs.zip"\
    "https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.9.0/Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE/Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/1.9.0/Xenium_V1_hSkin_nondiseased_section_1_FFPE/Xenium_V1_hSkin_nondiseased_section_1_FFPE_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/1.9.0/Xenium_V1_hLiver_nondiseased_section_FFPE/Xenium_V1_hLiver_nondiseased_section_FFPE_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/1.9.0/Xenium_V1_hHeart_nondiseased_section_FFPE/Xenium_V1_hHeart_nondiseased_section_FFPE_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_humanLung_Cancer_FFPE/Xenium_V1_humanLung_Cancer_FFPE_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/1.7.0/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hPancreas_Cancer_Add_on_FFPE/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hSkin_Melanoma_Base_FFPE/Xenium_V1_hSkin_Melanoma_Base_FFPE_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hColon_Non_diseased_Base_FFPE/Xenium_V1_hColon_Non_diseased_Base_FFPE_outs.zip"\
    "https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.6.0/Xenium_V1_mouse_pup/Xenium_V1_mouse_pup_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hLung_cancer_section/Xenium_V1_hLung_cancer_section_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hLymphNode_nondiseased_section/Xenium_V1_hLymphNode_nondiseased_section_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hPancreas_nondiseased_section/Xenium_V1_hPancreas_nondiseased_section_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hKidney_nondiseased_section/Xenium_V1_hKidney_nondiseased_section_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/1.4.0/Xenium_V1_FFPE_TgCRND8_17_9_months/Xenium_V1_FFPE_TgCRND8_17_9_months_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/1.3.0/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.zip"\
    "https://cf.10xgenomics.com/samples/xenium/1.3.0/Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon/Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs.zip"\
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
            echo "Donwloading $ZIP_REMOTE_PATH to $OUTPUT_DATASET_ZIP"
            curl $ZIP_REMOTE_PATH -o $OUTPUT_DATASET_ZIP
            echo "Successfully donwloaded"
        fi
        echo "Unzipping in $OUTPUT_DATASET_DIR"
        mkdir -p $OUTPUT_DATASET_DIR
        unzip -j $OUTPUT_DATASET_ZIP cell_feature_matrix.h5 cells.parquet -d $OUTPUT_DATASET_DIR
        echo "Successfully unzipped files"
    fi
done
