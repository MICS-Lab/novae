from pydantic import BaseModel


class DataConfig(BaseModel):
    train_dataset: str = "all"
    val_dataset: str | None = None


class PostTrainingConfig(BaseModel):
    save_umap: int | None = None
    save_result: bool = False
    save_metrics: bool = False
    log_domains: bool = False
    n_domains: list[int] = [7, 10]


class Config(BaseModel):
    project: str = "novae"
    wandb_artefact: str | None = None
    sweep: bool = False
    data: DataConfig = DataConfig()
    post_training: PostTrainingConfig = PostTrainingConfig()

    model_kwargs: dict = {}
    fit_kwargs: dict = {}
    wandb_init_kwargs: dict = {}
