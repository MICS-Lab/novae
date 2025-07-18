<p align="center">
  <img src="https://raw.githubusercontent.com/MICS-Lab/novae/main/docs/assets/banner.png" alt="novae_banner" width="100%"/>
</p>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/novae.svg)](https://pypi.org/project/novae)
[![Downloads](https://static.pepy.tech/badge/novae)](https://pepy.tech/project/novae)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://mics-lab.github.io/novae)
![Build](https://github.com/MICS-Lab/novae/workflows/ci/badge.svg)
[![License](https://img.shields.io/pypi/l/novae.svg)](https://github.com/MICS-Lab/novae/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/MICS-Lab/novae/graph/badge.svg?token=FFI44M52O9)](https://codecov.io/gh/MICS-Lab/novae)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

<p align="center"><b><i>
  💫 Graph-based foundation model for spatial transcriptomics data
</b></i></p>

Novae is a deep learning model for spatial domain assignments of spatial transcriptomics data (at both single-cell or spot resolution). It works across multiple gene panels, tissues, and technologies. Novae offers several additional features, including: (i) native batch-effect correction, (ii) analysis of spatially variable genes and pathways, and (iii) architecture analysis of tissue slides.

> [!NOTE]
> Novae was developed by the authors of [`sopa`](https://github.com/gustaveroussy/sopa) and is part of the [`scverse`](https://scverse.org/) ecosystem.

## Documentation

Check [Novae's documentation](https://mics-lab.github.io/novae/) to get started. It contains installation explanations, API details, and tutorials.

## Overview

<p align="center">
  <img src="https://raw.githubusercontent.com/MICS-Lab/novae/main/docs/assets/Figure1.png" alt="novae_overview" width="100%"/>
</p>

> **(a)** Novae was trained on a large dataset, and is shared on [Hugging Face Hub](https://huggingface.co/collections/MICS-Lab/novae-669cdf1754729d168a69f6bd). **(b)** Illustration of the main tasks and properties of Novae. **(c)** Illustration of the method behind Novae (self-supervision on graphs, adapted from [SwAV](https://arxiv.org/abs/2006.09882)).

## Installation

`novae` can be installed via `PyPI` on all OS, for any Python version from `3.10` to `3.12` (included).

```
pip install novae
```

> [!NOTE]
> See this [installation section](https://mics-lab.github.io/novae/getting_started/) for more details about extras and other installations modes.

## Usage

Here is a minimal usage example. For more details, refer to the [documentation](https://mics-lab.github.io/novae/).

```python
import novae

model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")

model.compute_representations(adata, zero_shot=True)
model.assign_domains(adata)
```

## Cite us

You can cite our [preprint](https://www.biorxiv.org/content/10.1101/2024.09.09.612009v1) as below:

```txt
@article{blampeyNovae2024,
  title = {Novae: A Graph-Based Foundation Model for Spatial Transcriptomics Data},
  author = {Blampey, Quentin and Benkirane, Hakim and Bercovici, Nadege and Andre, Fabrice and Cournede, Paul-Henry},
  year = {2024},
  pages = {2024.09.09.612009},
  publisher = {bioRxiv},
  doi = {10.1101/2024.09.09.612009},
}
```
