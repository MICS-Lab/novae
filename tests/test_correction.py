import numpy as np
import pandas as pd
from anndata import AnnData

import novae
from novae._constants import Keys

obs1 = pd.DataFrame(
    {
        Keys.SLIDE_ID: ["a", "a", "b", "b", "a", "b", "b"],
        "domains_key": ["N1", "N1", "N1", "N1", "N3", "N1", "N2"],
        Keys.IS_VALID_OBS: [True, True, True, True, True, True, True],
    },
    index=[f"{i}_1" for i in range(7)],
)

latent1 = np.array(
    [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [12, 13],
        [-2, -2],
    ]
).astype(np.float32)

expected1 = np.array(
    [
        [7, 8],
        [9, 10],
        [5, 6],
        [7, 8],
        [9, 10],
        [12, 13],
        [-2 + 35, -2 + 34],
    ]
).astype(np.float32)

adata1 = AnnData(obs=obs1, obsm={Keys.REPR: latent1})

obs2 = pd.DataFrame(
    {
        Keys.SLIDE_ID: ["c", "c", "c", "c", "c"],
        "domains_key": ["N2", "N1", np.nan, "N2", "N2"],
        Keys.IS_VALID_OBS: [True, True, False, True, True],
    },
    index=[f"{i}_2" for i in range(5)],
)

latent2 = np.array(
    [
        [-1, -3],
        [0, -10],
        [0, 0],
        [0, -1],
        [100, 100],
    ]
).astype(np.float32)

expected2 = np.array(
    [
        [-1, -3],
        [0 + 8, -10 + 19],
        [0, 0],
        [0, -1],
        [100, 100],
    ]
).astype(np.float32)

adata2 = AnnData(obs=obs2, obsm={Keys.REPR: latent2})

adatas = [adata1, adata2]
for adata in adatas:
    for key in [Keys.SLIDE_ID, "domains_key"]:
        adata.obs[key] = adata.obs[key].astype("category")


def test_batch_effect_correction():
    novae.utils.batch_effect_correction(adatas, "domains_key")
    assert (adata1.obsm[Keys.REPR_CORRECTED] == expected1).all()
    assert (adata2.obsm[Keys.REPR_CORRECTED] == expected2).all()
