import novae

adatas = novae.utils.dummy_dataset(xmax=200)

model = novae.Novae(adatas)

model.mode.trained = True
model.compute_representations()
model.assign_domains()


def test_plot_domains_hierarchy():
    model.plot_domains_hierarchy(hline_level=None)
    model.plot_domains_hierarchy(hline_level=3)
    model.plot_domains_hierarchy(hline_level=[2, 4])


def test_plot_prototype_weights():
    model.plot_prototype_weights()


def test_plot_domains():
    novae.plot.domains(adatas)
