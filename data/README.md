# Public datasets

We detail below how to download public spatial transcriptomics datasets. The data will be saved in this directory, and will be used to train `novae`.

## Download

For consistency, all the scripts below need to be executed at the root of the `data` directory (i.e., `novae/data`).

### MERSCOPE (18 samples)

Requirements: the `gsutil` command line should be installed (see [here](https://cloud.google.com/storage/docs/gsutil_install)) and a Python environment with `scanpy`.

```sh
# download all MERSCOPE datasets
sh _scripts/merscope_download.sh

# convert all datasets to h5ad files
python _scripts/merscope_convert.py
```

### Xenium (20+ samples)

Requirements: a Python environment with `spatialdata-io` installed.

```sh
# download all Xenium datasets
sh _scripts/xenium_download.sh

# convert all datasets to h5ad files
python _scripts/xenium_convert.py
```

### CosMX (3 samples)

Requirements: a Python environment with `scanpy` installed.

```sh
# download all CosMX datasets
sh _scripts/cosmx_download.sh

# convert all datasets to h5ad files
python _scripts/cosmx_convert.py
```

### All datasets

All above datasets can be downloaded using a single command line. Make sure you have all the requirements listed above.

```sh
sh _scripts/1_download_all.sh
```

## Preprocess and prepare for training

The script bellow will copy all `adata.h5ad` files into a single directory, compute UMAPs, and minor preprocessing. See the `argparse` helper of this script for more details.

```sh
python _scripts/2_prepare.py
```

## Usage

These datasets can be used during training (see the `scripts` directory at the root of the `novae` repository).

## Notes
- Missing technologies: CosMX, Curio Seeker, Resolve
- Public institute datasets with [STOmics DB](https://db.cngb.org/stomics/)
- Some Xenium datasets are available outside of the main "10X Datasets" page:
  - https://www.10xgenomics.com/products/visium-hd-spatial-gene-expression/dataset-human-crc
