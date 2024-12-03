## [0.2.0] - xxxx-xx-xx

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
