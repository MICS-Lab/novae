# Downloading public data

This directory contains `shell` and `python` scripts used to download public spatial transcriptomics datasets.

## Scripts

### MERSCOPE

Requirements: the `gsutil` command line should be installed. See [here](https://cloud.google.com/storage/docs/gsutil_install).

At the root of the `data` directory, run the following commands:

```sh
# download all MERSCOPE datasets
sh download/merscope_download.sh

# convert all datasets to h5ad files
python download/merscope_convert.py
```

### Xenium

Requirements: a Python environment with `spatialdata-io` installed.

At the root of the `data` directory, run the following commands:

```sh
# download all MERSCOPE datasets
sh download/xenium_download.sh

# convert all datasets to h5ad files
python download/xenium_convert.py
```
