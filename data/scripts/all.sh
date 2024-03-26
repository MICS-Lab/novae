#!/bin/bash

# download all MERSCOPE datasets
sh download/merscope_download.sh

# convert all datasets to h5ad files
python download/merscope_convert.py

# download all Xenium datasets
sh download/xenium_download.sh

# convert all datasets to h5ad files
python download/xenium_convert.py
