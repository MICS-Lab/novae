from typing import TYPE_CHECKING

from anndata import AnnData

if TYPE_CHECKING:
    from spatialdata import SpatialData

from .._constants import Keys, Nums


def compute_he_embeddings(
    sdata: "SpatialData",
    model: str,
    table_key: str = "table",
    patch_overlap_ratio: float = 0.5,
    image_key: str | None = None,
    device: str | None = None,
    batch_size: int = 32,
):
    try:
        import sopa
        from sopa._constants import SopaAttrs, SopaKeys
        from spatialdata.models import get_table_keys
    except ImportError:
        raise ImportError("Please install the multimodal extra via `pip install novae[multimodal]`.")

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

    sopa.patches.compute_embeddings(
        sdata,
        model,
        level=0,
        patch_width=Nums.HE_PATCH_WIDTH,
        patch_overlap=patch_overlap,
        image_key=image_key,
        device=device,
        batch_size=batch_size,
    )

    patches_centroids = sdata[SopaKeys.EMBEDDINGS_PATCHES].centroid
    closest_patch_index = patches_centroids.sindex.nearest(cells.centroid, return_all=False)[1]

    adata.obsm[Keys.HISTO_EMBEDDINGS] = sdata.tables[f"{model}_embeddings"].X[closest_patch_index]

    return adata
