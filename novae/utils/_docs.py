_auto_doc_map = {
    "adata": "An `AnnData` object, or a list of `AnnData` objects. Optional if the model was initialized with `adata`.",
    "slide_key": "Optional key of `adata.obs` containing the ID of each slide. Not needed if each `adata` is a slide.",
    "scgpt_model_dir": "Path to a directory containing a scGPT checkpoint, i.e. a `vocab.json` and a `best_model.pt` file.",
}

_format_map_dict = {k: f"{k}: {v}" for k, v in _auto_doc_map.items()}


def format_docstring(docstring: str) -> str:
    return docstring.format_map(_format_map_dict)


def doc_params(obj):
    obj.__doc__ = format_docstring(obj.__doc__)
    return obj
