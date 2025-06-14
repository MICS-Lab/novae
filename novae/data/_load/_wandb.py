import logging
from pathlib import Path

from ...utils import wandb_log_dir

log = logging.getLogger(__name__)


def _load_wandb_artifact(name: str) -> Path:
    import wandb

    api = wandb.Api()

    if not name.startswith("novae/"):
        name = f"novae/novae/{name}"

    artifact = api.artifact(name)

    artifact_path = wandb_log_dir() / "artifacts" / artifact.name

    if artifact_path.exists():
        log.info(f"Artifact {artifact_path} already downloaded")
    else:
        log.info(f"Downloading artifact at {artifact_path}")
        artifact.download(root=artifact_path)

    return artifact_path
