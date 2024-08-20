import pytest
import torch

import novae


def test_inference():
    adatas = novae.utils.dummy_dataset(xmax=200)

    model = novae.Novae(adatas)
    model._datamodule = model._init_datamodule()
    model.mode.trained = True

    with pytest.raises(AssertionError):
        model.infer_gene_expression(None, "CD4")

    before_params = [param.clone() for param in model.parameters()]

    model.fit_inference_head(max_epochs=1)

    with pytest.raises(AssertionError):
        model.infer_gene_expression(None, "CD4")

    model.compute_representations()
    model.infer_gene_expression(None, "CD4")
    model.infer_gene_expression(None, ["CD4", "CCL5"])

    with pytest.raises(ValueError):
        model.infer_gene_expression(None, "UNKNOWN_GENE_NAME")

    after_params = [param.clone() for param in model.parameters()]

    for before, after in zip(before_params, after_params):
        assert torch.equal(before, after), "Model parameters changed after the function call"
