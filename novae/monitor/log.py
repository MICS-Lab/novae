import io
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import wandb
from PIL import Image

log = logging.getLogger(__name__)


def repository_root() -> Path:
    """Get the path to the root of the repository (dev-mode users only)

    Returns:
        `novae` repository path
    """
    path = Path(__file__).parents[2]

    if path.name != "novae":
        log.warning("Can't find the novae repository path. Using the home directory instead.")
        return Path.home()

    return path


def wandb_log_dir() -> Path:
    return repository_root() / "wandb"


def log_plt_figure(name: str, dpi: int = 300) -> None:
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png", bbox_inches="tight", dpi=dpi)
    wandb.log({name: wandb.Image(Image.open(img_buf))})
    plt.close()


def wandb_results_dir() -> Path:
    res_dir: Path = wandb_log_dir() / "results" / wandb.run.name
    res_dir.mkdir(parents=True, exist_ok=True)
    return res_dir


def save_pdf_figure(name: str):
    plt.savefig(wandb_results_dir() / f"{name}.pdf", format="pdf", bbox_inches="tight")
