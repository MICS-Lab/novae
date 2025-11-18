import dask.array as da
import numpy as np
from fast_array_utils import stats
from fast_array_utils.conv import to_dense
from scipy.sparse import csr_matrix

x_np = np.array([
    [1, 0, 0, 3],
    [0, 5, 6, 0],
    [7, 0, -1, 0],
])

x_sparse = csr_matrix(x_np)

x_dask = da.from_array(x_np)


def test_backends_mean_var():
    mean_np, std_np = np.mean(x_np, axis=0), np.std(x_np, axis=0)

    fau_mean_np, fau_var_np = stats.mean_var(x_np, axis=0)
    fau_mean_sparse, fau_var_sparse = stats.mean_var(x_sparse, axis=0)
    fau_mean_dask, fau_var_dask = stats.mean_var(x_dask, axis=0)

    assert np.allclose(mean_np, fau_mean_np)
    assert np.allclose(std_np, fau_var_np**0.5)
    assert np.allclose(mean_np, fau_mean_sparse)
    assert np.allclose(std_np, fau_var_sparse**0.5)
    assert np.allclose(mean_np, fau_mean_dask)
    assert np.allclose(std_np, fau_var_dask**0.5)


def test_backends_min_max():
    min_np, max_np = np.min(x_np, axis=0), np.max(x_np, axis=0)

    fau_min_np = stats.min(x_np, axis=0)
    fau_min_sparse = stats.min(x_sparse, axis=0)
    fau_min_dask = stats.min(x_dask, axis=0)
    fau_max_np = stats.max(x_np, axis=0)
    fau_max_sparse = stats.max(x_sparse, axis=0)
    fau_max_dask = stats.max(x_dask, axis=0)

    assert np.allclose(min_np, fau_min_np)
    assert np.allclose(max_np, fau_max_np)
    assert np.allclose(min_np, fau_min_sparse)
    assert np.allclose(max_np, fau_max_sparse)
    assert np.allclose(min_np, fau_min_dask)
    assert np.allclose(max_np, fau_max_dask)


def test_spatially_variable_genes_backend():
    assert (to_dense(stats.mean(x_np > 0, axis=0), to_cpu_memory=True) > 0.5).sum() == 1
    assert (to_dense(stats.mean(x_sparse > 0, axis=0), to_cpu_memory=True) > 0.5).sum() == 1
