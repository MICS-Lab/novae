# Pancreas
PANCREAS_FLAT_FILES="https://smi-public.objects.liquidweb.services/cosmx-wtx/Pancreas-CosMx-WTx-FlatFiles.zip"
PANCREAS_OUTPUT_ZIP="cosmx/pancreas/Pancreas-CosMx-WTx-FlatFiles.zip"

mkdir -p cosmx/pancreas

if [ -f $PANCREAS_OUTPUT_ZIP ]; then
    echo "File $PANCREAS_OUTPUT_ZIP already exists."
else
    echo "Downloading $PANCREAS_FLAT_FILES to $PANCREAS_OUTPUT_ZIP"
    curl $PANCREAS_FLAT_FILES -o $PANCREAS_OUTPUT_ZIP
    unzip $PANCREAS_OUTPUT_ZIP -d cosmx/pancreas
fi

# Normal Liver
mkdir -p cosmx/normal_liver
METADATA_FILE="https://nanostring.app.box.com/index.php?rm=box_download_shared_file&shared_name=id16si2dckxqqpilexl2zg90leo57grn&file_id=f_1392279064291"
METADATA_OUTPUT="cosmx/normal_liver/metadata_file.csv"
if [ -f $METADATA_OUTPUT ]; then
    echo "File $METADATA_OUTPUT already exists."
else
    echo "Downloading $METADATA_FILE to $METADATA_OUTPUT"
    curl $METADATA_FILE -o $METADATA_OUTPUT
fi
COUNT_FILE="https://nanostring.app.box.com/index.php?rm=box_download_shared_file&shared_name=id16si2dckxqqpilexl2zg90leo57grn&file_id=f_1392318918584"
COUNT_OUTPUT="cosmx/normal_liver/exprMat_file.csv"
if [ -f $COUNT_OUTPUT ]; then
    echo "File $COUNT_OUTPUT already exists."
else
    echo "Downloading $COUNT_FILE to $COUNT_OUTPUT"
    curl $COUNT_FILE -o $COUNT_OUTPUT
fi

# Cancer Liver
mkdir -p cosmx/cancer_liver
METADATA_FILE="https://nanostring.app.box.com/index.php?rm=box_download_shared_file&shared_name=id16si2dckxqqpilexl2zg90leo57grn&file_id=f_1392293795557"
METADATA_OUTPUT="cosmx/cancer_liver/metadata_file.csv"
if [ -f $METADATA_OUTPUT ]; then
    echo "File $METADATA_OUTPUT already exists."
else
    echo "Downloading $METADATA_FILE to $METADATA_OUTPUT"
    curl $METADATA_FILE -o $METADATA_OUTPUT
fi
COUNT_FILE="https://nanostring.app.box.com/index.php?rm=box_download_shared_file&shared_name=id16si2dckxqqpilexl2zg90leo57grn&file_id=f_1392441469377"
COUNT_OUTPUT="cosmx/cancer_liver/exprMat_file.csv"
if [ -f $COUNT_OUTPUT ]; then
    echo "File $COUNT_OUTPUT already exists."
else
    echo "Downloading $COUNT_FILE to $COUNT_OUTPUT"
    curl $COUNT_FILE -o $COUNT_OUTPUT
fi
