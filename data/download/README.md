# Downloading public data

This directory contains `shell` and `python` scripts used to download public spatial transcriptomics datasets.

## Scripts

For conveniency, all the scripts below need to be executed at the root of the `data` directory.

### MERSCOPE (18 samples)

Requirements: the `gsutil` command line should be installed. See [here](https://cloud.google.com/storage/docs/gsutil_install).

```sh
# download all MERSCOPE datasets
sh download/merscope_download.sh

# convert all datasets to h5ad files
python download/merscope_convert.py
```

### Xenium (17 samples)

Requirements: a Python environment with `spatialdata-io` installed.

```sh
# download all Xenium datasets
sh download/xenium_download.sh

# convert all datasets to h5ad files
python download/xenium_convert.py
```

## Notes
- Missing technologies: CosMX, Curio Seeker, Resolve
- Public institute datasets with [STOmics DB](https://db.cngb.org/stomics/)
