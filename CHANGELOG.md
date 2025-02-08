## [0.2.3] - xxxx-xx-xx

### Fixed
- Fix `model.plot_prototype_covariance` and `model.plot_prototype_weights`
- Edge case: ensure that even negative max-weights are above the prototype threshold

### Added
- Infer default plot size for plots (median neighbors distrance)

### Changed
- `novae.utils.spatial_neighbors` can be now called via `novae.spatial_neighbors`

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
