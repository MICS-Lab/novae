import novae

from ._utils import adata_small as adata


def test_init_prototypes():
    model = novae.Novae(adata, num_prototypes=20)

    prototypes = model.swav_head.prototypes.data.clone()
    model.init_prototypes(adata)
    assert (model.swav_head.prototypes.data != prototypes).all()


def test_preserve_clustering():
    model = novae.Novae(adata)
    model.mode.trained = True

    clusters_levels = model.swav_head.clusters_levels
    leiden_codes = model._leiden_prototypes()

    model.save_pretrained("tests/test_proto_clustering")

    model2 = novae.Novae.from_pretrained("tests/test_proto_clustering")

    assert (model2.swav_head.clusters_levels == clusters_levels).all()
    assert (model2._leiden_prototypes() == leiden_codes).all()
