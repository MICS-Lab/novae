import numpy as np

from ._constants import Nums


class Settings:
    # misc settings
    auto_preprocessing: bool = True
    disable_multimodal: bool = False
    scale_to_microns: float = 1.0

    def disable_lazy_loading(self):
        """Disable lazy loading of subgraphs in the NovaeDataset."""
        Nums.LAZY_LOADING_SIZE_THRESHOLD = np.inf

    def enable_lazy_loading(self, lazy_loading_size_threshold: int = 0):
        """Enable lazy loading of subgraphs in the NovaeDataset.

        Args:
            lazy_loading_size_threshold: Lazy loading is used when the input has more elements (i.e., `n_obs * n_vars`) than this number.
        """
        Nums.LAZY_LOADING_SIZE_THRESHOLD = lazy_loading_size_threshold

    @property
    def warmup_epochs(self):
        return Nums.WARMUP_EPOCHS

    @warmup_epochs.setter
    def warmup_epochs(self, value: int):
        Nums.WARMUP_EPOCHS = value


settings = Settings()
