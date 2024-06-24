import novae

adatas = novae.utils.dummy_dataset(
    n_panels=2,
    n_slides_per_panel=2,
    n_obs_per_domain=100,
    n_domains=2,
)


def test_train():
    model = novae.Novae(adatas)
    model.train(max_epochs=1)
    model.latent_representation()
    model.assign_domains()
