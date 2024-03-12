from __future__ import annotations

import pytorch_lightning as pl
import scanpy as sc
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from graph_lr import GraphCL


def main():
    # adata = sc.read_h5ad("/Users/quentinblampey/dev/graph_lr/exploration/conc.h5ad")
    adata = sc.read_h5ad(
        "/Users/quentinblampey/data/vizgen/results/santiago_tumour_global_annot_epoch75_integrated.h5ad"
    )
    adata = adata[adata.obs.ID == "Santiago_2_region_2"].copy()

    swav = True
    slide_key = None  # "ID"

    mode = "swav" if swav else "shuffle"
    model = GraphCL(adata, swav, n_hops=2, heads=4, slide_key=slide_key)

    wandb_logger = WandbLogger(log_model="all", project=f"graph_lr_{mode}")

    callbacks = [ModelCheckpoint(monitor="loss_epoch")]

    trainer = pl.Trainer(
        max_epochs=40,
        accelerator="cpu",
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
