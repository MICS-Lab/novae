import io

import matplotlib.pyplot as plt
import wandb
from PIL import Image


def log_plt_figure(name: str):

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png", bbox_inches="tight")
    wandb.log({name: wandb.Image(Image.open(img_buf))})
    plt.close()
