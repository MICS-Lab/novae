import numpy as np

from ._constants import Nums


class Settings:
    # misc settings
    auto_preprocessing: bool = True

    def disable_lazy_loading(self):
        """Disable lazy loading of subgraphs in the NovaeDataset."""
        Nums.N_OBS_THRESHOLD = np.inf

    def enable_lazy_loading(self, n_obs_threshold: int = 0):
        """Enable lazy loading of subgraphs in the NovaeDataset.

        Args:
            n_obs_threshold: Lazy loading is used above this number of cells in an AnnData object.
        """
        Nums.N_OBS_THRESHOLD = n_obs_threshold

    @property
    def warmup_epochs(self):
        return Nums.WARMUP_EPOCHS

    @warmup_epochs.setter
    def warmup_epochs(self, value: int):
        Nums.WARMUP_EPOCHS = value


settings = Settings()
