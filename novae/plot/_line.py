from pathlib import Path

import pandas as pd
import seaborn as sns


def loss_curve(log_dir: str, version: int = -1, on_epoch: bool = False, **kwargs: int) -> None:
    """Plot the training loss curve from the CSV logs. This is a basic alternative for monitoring training when Weights & Biases can't be used.

    !!! info
        To use this function, you need to fit or fine_tune Novae with a CSVLogger, for example:
        ```python
        from lightning.pytorch.loggers import CSVLogger

        # save the logs in a directory called "logs"
        model.fine_tune(logger=CSVLogger("logs"), log_every_n_steps=10)

        novae.plot.loss_curve("logs")
        ```

    Args:
        log_dir: The name of the directory containing the CSVLogger logs.
        version: Version of the run. By default, searches for the latest run.
        on_epoch: Whether to show the loss per epoch or per step.
        **kwargs: Additional keyword arguments passed to `seaborn.lineplot`.
    """
    log_dir: Path = Path(log_dir) / "lightning_logs"

    if version == -1:
        version = max(int(d.name.split("_")[-1]) for d in log_dir.iterdir() if d.is_dir())

    df = pd.read_csv(log_dir / f"version_{version}" / "metrics.csv")

    x = "epoch" if on_epoch else "step"
    y = f"train/loss_{x}"

    sns.lineplot(df[[x, y]].dropna(), x=x, y=y, **kwargs)
    sns.despine(offset=10, trim=True)
