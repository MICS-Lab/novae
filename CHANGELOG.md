## [1.0.2] - 2026-01-15

### Added
- Support saving/loading a model from a `Path` (before, only `str` was supported)

### Changed
- Cleanup: removed unused `pixel_size` argument from `novae.spatial_neighbors`

### Fixed
- Upgrade sopa to import `sopa.constants` instead of `sopa._constants` (#38)

## [1.0.1] - 2025-12-10

🎉 Novae is now [published in Nature Methods](https://www.nature.com/articles/s41592-025-02899-6)!

### Added
- Full dataset added to [our Hugging Face Hub](https://huggingface.co/datasets/MICS-Lab/novae)
- Official Docker images released - [see here](https://hub.docker.com/r/quentinblampey/novae)
- Support `reference=None` in `fine_tune` (random initialization) and used as default
- Added `fast` argument in `novae.plot.domains` to have a quick (but less accurate) rendering of domains.
- Added `novae.settings.scale_to_microns` if the coordinates are not in microns
- Use [`fast-array-utils`](https://github.com/scverse/fast-array-utils) to support multiple backends in `adata.X` (e.g., dask or backed mode)

### Breaking changes
- Remove support for `python==3.10`

### Changed
- Update hf-hub to use `xet` for faster dataset download from Hugging Face Hub
- Update to `spatialdata>=0.5.0` to avoid installation issue related to `xarray-dataclasses`
- Threshold to decide when to enable lazy loading: based on `n_obs * n_vars` (i.e., the array size) instead of `n_obs`

### Fixed
- Retrieve `mode=multimodal` when saving the model and re-loading the model after H&E training (#24)
- Fix tests for recent AnnData versions only supporting CSR/CSC sparse matrices

## [1.0.0] - 2025-08-09

Pre-publication release (for Zenodo DOI creation).

### Added
- Support multimodality (H&E + spatial omics). See [this tutorial](https://mics-lab.github.io/novae/tutorials/he_usage/).
- Support multi-references in `fine_tune` and zero-shot modes
- Added `novae.plot.loss_curve` for minimal monitoring when not using Weight & Biases

### Changed
- `fine_tune` method: use `lr = 5e-4` and `max_epochs = 20` as new default values
- Use `reference="all"` instead of `"largest"` by default
- Move `novae.utils.load_dataset` to `novae.load_dataset` (the old import is deprecated, it will be removed in future versions)
- Move `novae.utils.quantile_scaling` to `novae.quantile_scaling` (deprecated, as above)
- Move `novae.utils.toy_dataset` to `novae.toy_dataset` (deprecated, as above)
- Migrate to `uv` as a package manager, and `ruff` for formatting/linting

### Fixed
- Slide-id passed correctly to dataloader for one-adata multi-slide mode
- Auto-detect change in n_hops_{local,view} to re-build graph
- Move representations to `numpy` when torch is not needed anymore (#19)

## [0.2.4] - 2025-03-26

Hotfix (#18)

## [0.2.3] - 2025-03-21

### Added
- New Visium-HD and Visium tutorials
- Infer default plot size for plots (median neighbors distrance)
- Can filter by technology in `load_dataset`

### Fixed
- Fix `model.plot_prototype_covariance` and `model.plot_prototype_weights`
- Edge case: ensure that even negative max-weights are above the prototype threshold

### Changed
- `novae.utils.spatial_neighbors` can be now called via `novae.spatial_neighbors`
- Store distances after `spatial_neighbors` instead of `1` when `coord_type="grid"`

## [0.2.2] - 2024-12-17

### Added
- `load_dataset`: add `custom_filter` and `dry_run` arguments
- added `min_prototypes_ratio` argument in `fine_tune` to run `init_slide_queue`
- Added tutorials for proteins data + minor docs improvements

### Fixed
- Ensure reset clustering if multiple zero-shot (#9)

### Changed
- Removed the docs formatting (better for autocompletion)
- Reorder parameters in Novae `__init__` (sorted by importance)

## [0.2.1] - 2024-12-04

### Added
- `novae.utils.quantile_scaling` for proteins expression

### Fixed
- Fix autocompletion using `__new__` to trick hugging_face inheritance


## [0.2.0] - 2024-12-03

### Added

- `novae.plot.connectivities(...)` to show the cells neighbors
- `novae.settings.auto_processing = False` to enforce using your own preprocessing
- Tutorials update (more plots and more details)

### Fixed

- Issue with `library_id` in `novae.plot.domains` (#8)
- Set `pandas>=2.0.0` in the dependencies (#5)

### Breaking changes

- `novae.utils.spatial_neighbors` must always be run, to force the user having more control on it
- For multi-slide mode, the `slide_key` argument should now be used in `novae.utils.spatial_neighbors` (and only there)
- Drop python 3.9 support (because dropped in `anndata`)

## [0.1.0] - 2024-09-11

First official `novae` release. Preprint coming soon.
