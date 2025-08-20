import numpy as np
import pytest
from anndata import AnnData

from novae.utils._validate import _validate_preprocessing


def test_validate_preprocessing():
    adata = AnnData(np.array([[int(np.exp(11))]]))

    with pytest.raises(AssertionError):
        _validate_preprocessing([adata])  # expecting an error because the max value is too high

    adata = AnnData(np.array([[int(np.exp(9))]]))

    _validate_preprocessing([adata])  # should not raise an error
