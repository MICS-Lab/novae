import io
from pathlib import Path

import matplotlib.pyplot as plt
import wandb
from PIL import Image

from ..utils import repository_root


def log_plt_figure(name: str, dpi: int = 300) -> None:
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png", bbox_inches="tight", dpi=dpi)
    wandb.log({name: wandb.Image(Image.open(img_buf))})
    plt.close()


def save_pdf_figure(name: str):
    plt.savefig(wandb_results_dir() / f"{name}.pdf", format="pdf", bbox_inches="tight")


def wandb_results_dir() -> Path:
    res_dir: Path = repository_root() / "data" / "results" / wandb.run.name
    res_dir.mkdir(parents=True, exist_ok=True)
    return res_dir
