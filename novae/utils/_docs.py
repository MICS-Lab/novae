_format_doc_dict = {
    "adata": "An `AnnData` object, or a list of `AnnData` objects. Optional if the model was initialized with `adata`.",
    "slide_key": "Optional key of `adata.obs` containing the ID of each slide. Not needed if each `adata` is a slide.",
    "var_names": "Only used when loading a pretrained model. To not use it yourself.",
    "scgpt_model_dir": "Path to a directory containing a scGPT checkpoint, i.e. a `vocab.json` and a `best_model.pt` file.",
    "panel_subset_size": "Ratio of genes kept from the panel during augmentation.",
    "background_noise_lambda": "Parameter of the exponential distribution for the noise augmentation.",
    "sensitivity_noise_std": "Standard deviation for the multiplicative for for the noise augmentation.",
    "data": "A Pytorch Geometric `Data` object representing a batch of `B` graphs.",
    "obs_key": "Key of `adata.obs` containing the niches annotation.",
    "n_top_genes": "Number of genes per niche to consider.",
}

_format_doc_map = {k: f"{k}: {v}" for k, v in _format_doc_dict.items()}


def format_docstring(docstring: str) -> str:
    return docstring.format_map(_format_doc_map)


def format_docs(obj):
    obj.__doc__ = format_docstring(obj.__doc__)
    return obj
