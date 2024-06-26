# Novae

Graph-based foundation model for spatial transcriptomics data.

## Documentation

The [complete documentation can be found here](https://mics-lab.github.io/novae/). It contains installation guidelines, tutorials, a description of the API, etc.

## Overview

TODO

## Installation

### PyPI

`novae` can be installed via `PyPI` on all OS, for `python>=3.9`.

```
pip install novae
```

### Editable mode

To install `novae` in editable mode (e.g., to contribute), clone the repository and choose among the options below.

```sh
pip install -e .                 # pip, minimal dependencies
pip install -e '.[dev,monitor]'  # pip, all extras
poetry install                   # poetry, minimal dependencies
poetry install --all-extras      # poetry, all extras
```

## Usage

Here is a minimal usage guide. For more details (e.g. how to load a pretrained model), refer to the [documentation](https://mics-lab.github.io/novae/).

```python
import novae

model = novae.Novae(adata)

model.fit()
model.latent_representation()
model.assign_domains()
```

## Cite us

TODO
