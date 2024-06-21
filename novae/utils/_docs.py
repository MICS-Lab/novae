_format_doc_dict = {
    "adata": "An `AnnData` object, or a list of `AnnData` objects. Optional if the model was initialized with `adata`.",
    "slide_key": "Optional key of `adata.obs` containing the ID of each slide. Not needed if each `adata` is a slide.",
    "scgpt_model_dir": "Path to a directory containing a scGPT checkpoint, i.e. a `vocab.json` and a `best_model.pt` file.",
}

_format_doc_map = {k: f"{k}: {v}" for k, v in _format_doc_dict.items()}


def format_docstring(docstring: str) -> str:
    return docstring.format_map(_format_doc_map)


def format_docs(obj):
    obj.__doc__ = format_docstring(obj.__doc__)
    return obj
