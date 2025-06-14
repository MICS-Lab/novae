import logging
from typing import TYPE_CHECKING, Callable, Union

import numpy as np
from anndata import AnnData
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from spatialdata import SpatialData

from ..._constants import Keys, Nums

log = logging.getLogger(__name__)


def compute_histo_embeddings(
    sdata: "SpatialData",
    model: str | Callable = "conch",
    table_key: str = "table",
    patch_overlap_ratio: float = 0.5,
    image_key: str | None = None,
    device: str | None = None,
    batch_size: int = 32,
) -> None:
    """Compute histology embeddings for a given model on a grid of overlapping patches.

    It will add a new `AnnData` object to the `SpatialData` object, containing the embeddings of the patches, and
    add a column in the cells table with the index of the closest patch.

    !!! warning "Installation"
        This function requires the `multimodal` extra. You can install it with `pip install novae[multimodal]`.

    Args:
        sdata: A `SpatialData` object containing the data.
        model: The model to use for computing embeddings. See the [sopa documentation](https://gustaveroussy.github.io/sopa/api/patches/#sopa.patches.compute_embeddings) for more details.
        table_key: Name of the `AnnData` object containing the cells.
        patch_overlap_ratio: Ratio of overlap between patches.
        image_key: Name of the histology image. If None, the function will try to find the image key automatically.
        device: Torch device to use for computation.
        batch_size: Mini-batch size for computation.
    """
    try:
        import sopa
        from sopa._constants import SopaAttrs, SopaKeys
        from spatialdata.models import get_table_keys
    except ImportError:
        raise ImportError(
            "Please install the multimodal extra via `pip install novae[multimodal]`.\nIf you want to use CONCH, also install the corresponding extra via `pip install 'novae[multimodal,conch]'`."
        )

    assert 0 <= patch_overlap_ratio < 1, "patch_overlap_ratio must be between 0 and 1"
    patch_overlap = int(patch_overlap_ratio * Nums.HE_PATCH_WIDTH)

    adata: AnnData = sdata[table_key]

    if image_key is None:
        image_key, _ = sopa.utils.get_spatial_element(
            sdata.images, sdata.attrs.get(SopaAttrs.TISSUE_SEGMENTATION, None), return_key=True
        )

    # align the cells boundaries to the image coordinate system
    shapes_key, _, instance_key = get_table_keys(adata)
    cells = sopa.utils.to_intrinsic(sdata, shapes_key, image_key)
    cells = cells.loc[adata.obs[instance_key]]

    key_added = sopa.patches.compute_embeddings(
        sdata,
        model,
        level=0,
        patch_width=Nums.HE_PATCH_WIDTH,
        patch_overlap=patch_overlap,
        image_key=image_key,
        device=device,
        batch_size=batch_size,
        roi_key=shapes_key,
    )

    patches_centroids = sdata[SopaKeys.EMBEDDINGS_PATCHES].centroid
    indices, distances = patches_centroids.sindex.nearest(cells.centroid, return_all=False, return_distance=True)

    _quality_control_join(distances)

    adata.obs["embedding_key"] = key_added
    adata.obs["embedding_index"] = indices[1]


def _quality_control_join(distances: np.ndarray):
    ADVICE = "Consider increasing the `patch_overlap_ratio`, or check that no cell is out of the image."

    mean_distance = distances.mean()
    if mean_distance > Nums.HE_PATCH_WIDTH / 4:
        log.warning(f"The mean distance between patches and cells is {mean_distance:.2f}, which is high. {ADVICE}")

    ratio_cells_far = (distances > Nums.HE_PATCH_WIDTH).mean()
    if ratio_cells_far > 0.05:
        log.warning(f"More than {ratio_cells_far:.2%} of cells are far from their patches. {ADVICE}")


def compute_histo_pca(
    sdatas: Union["SpatialData", list["SpatialData"]], n_components: int = 50, table_key: str = "table"
) -> None:
    """Run PCA on the histology embeddings associated to each cell (from the closest patch).
    The embedding is stored in `adata.obsm["histo_embeddings"]`, where `adata` is the table of cell expression.

    !!! info
        You need to run [novae.data.compute_histo_embeddings][] before running this function.

    Args:
        sdatas: One or several `SpatialData` object(s).
        n_components: Number of components for the PCA.
        table_key: Name of the `AnnData` object containing the cells.
    """
    from spatialdata import SpatialData

    if isinstance(sdatas, SpatialData):
        sdatas = [sdatas]

    def _histo_emb(sdata: "SpatialData") -> np.ndarray:
        _table: AnnData = sdata.tables[table_key]

        assert "embedding_key" in _table.obs, (
            "Could not find `embedding_key` in adata.obs. Did you run `novae.data.compute_histo_embeddings` first?"
        )
        embedding_key = _table.obs["embedding_key"].iloc[0]
        return sdata.tables[embedding_key].X

    X = np.concatenate([_histo_emb(sdata) for sdata in sdatas], axis=0)

    pca = PCA(n_components=n_components)
    pipeline = Pipeline([("pca", pca), ("scaler", StandardScaler())])

    pipeline.fit(X)

    for sdata in sdatas:
        adata: AnnData = sdata[table_key]
        adata.obsm[Keys.HISTO_EMBEDDINGS] = pipeline.transform(_histo_emb(sdata)[adata.obs["embedding_index"]])
