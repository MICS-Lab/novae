# Public datasets

We detail below how to download public spatial transcriptomics datasets. The data will be saved in this directory, and will be used to train `novae`.

## Download

For consistency, all the scripts below need to be executed at the root of this directory (i.e., `novae/data`).

### MERSCOPE (18 samples)

Requirements: the `gsutil` command line should be installed. See [here](https://cloud.google.com/storage/docs/gsutil_install).

```sh
# download all MERSCOPE datasets
sh scripts/merscope_download.sh

# convert all datasets to h5ad files
python scripts/merscope_convert.py
```

### Xenium (17 samples)

Requirements: a Python environment with `spatialdata-io` installed.

```sh
# download all Xenium datasets
sh scripts/xenium_download.sh

# convert all datasets to h5ad files
python scripts/xenium_convert.py
```

## Usage

These datasets can be used during training (see the `scripts` directory).

## Notes
- Missing technologies: CosMX, Curio Seeker, Resolve
- Public institute datasets with [STOmics DB](https://db.cngb.org/stomics/)
