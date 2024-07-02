import pytest

import novae


def test_load_old_model():
    # should tell the model from the checkpoint was trained with an old novae version
    with pytest.raises(ValueError):
        novae.Novae.load_from_wandb_artifact("novae/novae_swav/model-w9xjvjjy:v29")


def test_load_model():
    novae.Novae.load_from_wandb_artifact("novae/novae/model-4i8e9g2v:v17")
