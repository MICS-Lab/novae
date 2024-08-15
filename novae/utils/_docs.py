_format_doc_dict = {
    "adata": "An `AnnData` object, or a list of `AnnData` objects. Optional if the model was initialized with `adata`.",
    "slide_key": "Optional key of `adata.obs` containing the ID of each slide. Not needed if each `adata` is a slide.",
    "var_names": "Only used when loading a pretrained model. To not use it yourself.",
    "scgpt_model_dir": "Path to a directory containing a scGPT checkpoint, i.e. a `vocab.json` and a `best_model.pt` file.",
    "panel_subset_size": "Ratio of genes kept from the panel during augmentation.",
    "background_noise_lambda": "Parameter of the exponential distribution for the noise augmentation.",
    "sensitivity_noise_std": "Standard deviation for the multiplicative for for the noise augmentation.",
    "data": "A Pytorch Geometric `Data` object representing a batch of `B` graphs.",
    "obs_key": "Key of `adata.obs` containing the domains annotation.",
    "n_top_genes": "Number of genes per domain to consider.",
    "n_hops_local": "Number of hops between a cell and its neighborhood cells.",
    "n_hops_view": "Number of hops between a cell and the origin of a second graph (or 'view').",
    "accelerator": "Accelerator to use. For instance, `'cuda'`, `'cpu'`, or `'auto'`. See Pytorch Lightning for more details.",
    "num_workers": "Number of workers for the dataloader.",
    "z": "The representations of one batch, of size `(B, O)`.",
    "projections": "Projections of the (normalized) representations over the prototypes, of size `(B, K)`.",
    "output_size": "Size of the representations, i.e. the encoder outputs (`O` in the article).",
    "num_prototypes": "Number of prototypes (`K` in the article).",
    "temperature": "Temperature used in the cross-entropy loss.",
    "embedding_size": "Size of the embeddings of the genes (`E` in the article).",
}

_format_doc_map = {k: f"{k}: {v}" for k, v in _format_doc_dict.items()}


def format_docstring(docstring: str) -> str:
    return docstring.format_map(_format_doc_map)


def format_docs(obj):
    obj.__doc__ = format_docstring(obj.__doc__)
    return obj
