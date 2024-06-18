#!/bin/bash

# download all MERSCOPE datasets
sh merscope_download.sh

# convert all datasets to h5ad files
python merscope_convert.py

# download all Xenium datasets
sh xenium_download.sh

# convert all datasets to h5ad files
python xenium_convert.py

# download all CosMX datasets
sh cosmx_download.sh

# convert all datasets to h5ad files
python cosmx_convert.py
