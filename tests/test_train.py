import numpy as np

import novae

adatas = novae.utils.dummy_dataset(
    n_panels=2,
    n_slides_per_panel=2,
    n_obs_per_domain=100,
    n_domains=2,
)


def test_train():
    model = novae.Novae(adatas)
    model.fit(max_epochs=1)
    model.latent_representation()
    obs_key = model.assign_domains(k=2)

    adatas[0].obs.iloc[0][obs_key] = np.nan

    svg1 = novae.monitor.mean_svg_score(adatas, obs_key=obs_key)
    fide1 = novae.monitor.mean_fide_score(adatas, obs_key=obs_key)
    jsd1 = novae.monitor.jensen_shannon_divergence(adatas, obs_key=obs_key)

    svg2 = novae.monitor.mean_svg_score(adatas, obs_key=obs_key, slide_key="slide_key")
    fide2 = novae.monitor.mean_fide_score(adatas, obs_key=obs_key, slide_key="slide_key")
    jsd2 = novae.monitor.jensen_shannon_divergence(adatas, obs_key=obs_key, slide_key="slide_key")

    assert svg1 != svg2
    assert fide1 != fide2
    assert jsd1 != jsd2
