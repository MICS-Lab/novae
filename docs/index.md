<p align="center">
  <img src="./assets/logo_white.png" alt="novae_logo" width="300px"/>
</p>

<p align="center"><b><i>
  💫 Graph-based foundation model for spatial transcriptomics data
</b></i></p>

Novae is a deep learning model for spatial domain assignments of spatial transcriptomics data (at both single-cell or spot resolution). It works across multiple gene panels, tissues, and technologies. Novae offers several additional features, including: (i) native batch-effect correction, (ii) analysis of spatially variable genes and pathways, and (iii) architecture analysis of tissue slides.

!!! info
    Novae was developed by the authors of [`sopa`](https://github.com/gustaveroussy/sopa) and is part of the [`scverse`](https://scverse.org/) ecosystem.

## Overview

<p align="center">
  <img src="https://raw.githubusercontent.com/MICS-Lab/novae/main/docs/assets/Figure1.png" alt="novae_overview" width="100%"/>
</p>

> **(a)** Novae was trained on a large dataset, and is shared on [Hugging Face Hub](https://huggingface.co/collections/MICS-Lab/novae-669cdf1754729d168a69f6bd). **(b)** Illustration of the main tasks and properties of Novae. **(c)** Illustration of the method behind Novae (self-supervision on graphs, adapted from [SwAV](https://arxiv.org/abs/2006.09882)).


## Why using Novae

- It is already pretrained on a large dataset (pan human/mouse tissues, brain, ...). Therefore, you can compute spatial domains in a zero-shot manner or quickly fine-tune the model.
- It has been developed to find consistent domains across many slides. This also works if you have different technologies (e.g., MERSCOPE/Xenium) and multiple gene panels.
- You can natively correct the spatial-domains batch-effect, without using external tools.
- After inference, the spatial domain assignment is super fast, allowing you to try multiple resolutions easily.
- It supports many downstream tasks, all included inside one framework.
