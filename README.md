<p align="center">
  <img src="docs/assets/banner.png" alt="novae_banner" width="100%"/>
</p>
<!-- TODO: when it becomes public: https://raw.githubusercontent.com/MICS-Lab/novae/main/docs/assets/banner.png -->

<p align="center"><b><i>
  💫 Graph-based foundation model for spatial transcriptomics data
</b></i></p>

## Documentation

Check [Novae's documentation](https://mics-lab.github.io/novae/) to get started. It contains installation explanations, API details, and tutorials.

## Overview

TODO
<!-- TODO: when it becomes public: update image link -->

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

model = novae.Novae.from_pretrained("...")

model.compute_representation()
model.assign_domains()
```

## Cite us

```txt
TODO
```
