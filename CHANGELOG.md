## [1.0.0] - xxxx-xx-xx

First post-publication release.
Starting from this version, it will be as backward compatible as possible.

## Changed
- Move `novae.utils.load_dataset` to `novae.load_dataset`
- Move `novae.utils.quantile_scaling` to `novae.data.quantile_scaling`
- Move `novae.utils.toy_dataset` to `novae.data.toy_dataset`
- Migrate to `uv` + `ruff`
- Move representations to `numpy` when torch is not needed anymore

## Fixed
- Removed scaling in the data loader

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
