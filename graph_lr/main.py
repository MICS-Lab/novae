import pytorch_lightning as pl
import scanpy as sc
from pytorch_lightning.loggers import WandbLogger

from graph_lr import GraphCL


def main():
    adata = sc.read_h5ad("/Users/quentinblampey/dev/graph_lr/exploration/conc.h5ad")

    model = GraphCL(adata, n_hops=2, heads=4)

    wandb_logger = WandbLogger(log_model="all", project="graph-lr")

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="cpu",
        enable_checkpointing=False,
        log_every_n_steps=10,
        logger=wandb_logger,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()