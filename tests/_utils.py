import anndata
import numpy as np
import pandas as pd
from anndata import AnnData

import novae

domains = [
    ["D1", "D2", "D3", "D4", "D5"],
    ["D1", "D1", "D2", "D2", "D3"],
    ["D1", "D1", "D1", "D2", np.nan],
]

spatial_coords = np.array(
    [
        [[0, 0], [0, 1], [0, 2], [3, 3], [1, 1]],
        [[0, 0], [0, 1], [0, 2], [3, 3], [1, 1]],
        [[0, 0], [0, 1], [0, 2], [3, 3], [1, 1]],
    ]
)


def _get_adata(i: int):
    values = domains[i]
    adata = AnnData(obs=pd.DataFrame({"domain": values}, index=[str(i) for i in range(len(values))]))
    adata.obs["slide_key"] = f"slide_{i}"
    adata.obsm["spatial"] = spatial_coords[i]
    novae.utils.spatial_neighbors(adata, radius=[0, 1.5])
    return adata


adatas = [_get_adata(i) for i in range(len(domains))]

adata = adatas[0]

adata_concat = anndata.concat(adatas)

# o
#
# o-o
#
# o-o-o
spatial_coords2 = np.array(
    [[0, 0], [0, 1], [0, 2], [2, 0], [2, 1], [4, 0]],
)

adata_line = AnnData(obs=pd.DataFrame(index=[str(i) for i in range(len(spatial_coords2))]))
adata_line.obsm["spatial"] = spatial_coords2
novae.utils.spatial_neighbors(adata_line, radius=1.5)
