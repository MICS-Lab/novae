import novae
from novae._constants import Keys
from novae.utils import get_reference


def test_get_reference():
    adatas = novae.toy_dataset()

    novae.spatial_neighbors(adatas)

    assert get_reference(adatas, "largest").n_obs == 1958

    assert len(get_reference(adatas, "all")) == 3

    assert get_reference(adatas, 1)[0].n_obs == 1957

    assert [adata.n_obs for adata in get_reference(adatas, [1, 2])] == [1957, 1956]

    sid1 = adatas[1].obs[Keys.SLIDE_ID].iloc[0]
    assert get_reference(adatas, sid1)[0].obs[Keys.SLIDE_ID].iloc[0] == sid1
