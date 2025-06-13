from anndata import AnnData

from .. import settings
from .._constants import Keys


class Mode:
    """Novae mode class, used to store states variables related to training and inference."""

    zero_shot_clustering_attrs: list[str] = ["_clustering_zero", "_clusters_levels_zero"]
    normal_clustering_attrs: list[str] = ["_clustering", "_clusters_levels"]
    all_clustering_attrs: list[str] = normal_clustering_attrs + zero_shot_clustering_attrs

    def __init__(self):
        self.zero_shot = False
        self.trained = False
        self.pretrained = False
        self.multimodal = False

    def __repr__(self) -> str:
        return f"Mode({dict(self.__dict__.items())})"

    ### Mode modifiers

    def from_pretrained(self):
        self.zero_shot = False
        self.trained = True
        self.pretrained = True

    def fine_tune(self):
        assert self.pretrained, "Fine-tuning requires a pretrained model."
        self.zero_shot = False

    def fit(self):
        self.zero_shot = False
        self.trained = False

    def update_multimodal_mode(self, adata: AnnData | list[AnnData] | None):
        if adata is None or settings.disable_multimodal:
            return

        adata = adata if isinstance(adata, AnnData) else adata[0]

        self.multimodal = Keys.HISTO_EMBEDDINGS in adata.obsm

    ### Mode-specific attributes

    @property
    def clustering_attr(self):
        return "_clustering_zero" if self.zero_shot else "_clustering"

    @property
    def clusters_levels_attr(self):
        return "_clusters_levels_zero" if self.zero_shot else "_clusters_levels"
