import numpy as np
import pytest
from anndata import AnnData

from novae.utils._validate import _standardize_adatas


def test_standardize_adatas():
    adata = AnnData(np.array([[int(np.exp(11))]]))

    with pytest.raises(AssertionError):
        _standardize_adatas([adata])  # expecting an error because the max value is too high

    adata = AnnData(np.array([[int(np.exp(9))]]))

    _standardize_adatas([adata])  # should not raise an error
