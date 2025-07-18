# (optional) create a new directory
mkdir Xenium_Prime_Human_Lung_Cancer_FFPE_outs
cd Xenium_Prime_Human_Lung_Cancer_FFPE_outs

# download H&E/alignment files
curl -O https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Human_Lung_Cancer_FFPE/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image.ome.tif
curl -O https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Human_Lung_Cancer_FFPE/Xenium_Prime_Human_Lung_Cancer_FFPE_he_imagealignment.csv

# download and unzip xenium output files
curl -O https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/3.0.0/Xenium_Prime_Human_Lung_Cancer_FFPE/Xenium_Prime_Human_Lung_Cancer_FFPE_outs.zip
unzip Xenium_Prime_Human_Lung_Cancer_FFPE_outs.zip
