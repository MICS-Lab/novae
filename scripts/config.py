from __future__ import annotations

from pydantic import BaseModel


class DataConfig(BaseModel):
    train_dataset: str = "all"
    val_dataset: str | None = None

    files_black_list: list[str] = []


class PostTrainingConfig(BaseModel):
    n_domains: list[int] = [7, 10]
    log_umap: bool = False
    log_metrics: bool = False
    log_domains: bool = False
    save_h5ad: bool = False
    delete_X: bool = False


class Config(BaseModel):
    project: str = "novae"
    wandb_artefact: str | None = None
    zero_shot: bool = False
    sweep: bool = False

    data: DataConfig = DataConfig()
    post_training: PostTrainingConfig = PostTrainingConfig()

    model_kwargs: dict = {}
    fit_kwargs: dict = {}
    wandb_init_kwargs: dict = {}
