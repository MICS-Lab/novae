from __future__ import annotations

import pytest

import novae
from novae._constants import Keys

adatas = novae.utils.dummy_dataset(n_panels=2, xmax=200)

model = novae.Novae(adatas)

model.mode.trained = True
model.compute_representations()
model.assign_domains()


@pytest.mark.parametrize("hline_level", [None, 3, [2, 4]])
def test_plot_domains_hierarchy(hline_level: int | list[int] | None):
    model.plot_domains_hierarchy(hline_level=hline_level)


def test_plot_prototype_weights():
    model.plot_prototype_weights()

    adatas[0].uns[Keys.UNS_TISSUE] = "breast"
    adatas[1].uns[Keys.UNS_TISSUE] = "lung"

    model.plot_prototype_weights()


def test_plot_domains():
    novae.plot.domains(adatas)