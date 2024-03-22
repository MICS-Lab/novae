### Copied from: https://github.com/zhaofangyuan98/SDMBench

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


def res_search(adata: AnnData, target_k=7, res_start=0.1, res_step=0.1, res_epochs=10):
    """
    adata: the Anndata object, a dataset.
    target_k: int, expected number of clusters.
    res_start: float, starting value of resolution. default: 0.1.
    res_step: float, step of resoution. default: 0.1.
    res_epochs: int, epoch of resolution. default: 10.
    """

    print(f"searching resolution to k={target_k}")
    res = res_start
    sc.tl.leiden(adata, resolution=res)

    old_k = len(adata.obs["leiden"].cat.categories)
    print("Res = ", res, "Num of clusters = ", old_k)

    run = 0
    while old_k != target_k:
        old_sign = 1 if (old_k < target_k) else -1
        sc.tl.leiden(adata, resolution=res + res_step * old_sign)
        new_k = len(adata.obs["leiden"].cat.categories)
        print("Res = ", res + res_step * old_sign, "Num of clusters = ", new_k)
        if new_k == target_k:
            res = res + res_step * old_sign
            print("recommended res = ", str(res))
            return res
        new_sign = 1 if (new_k < target_k) else -1
        if new_sign == old_sign:
            res = res + res_step * old_sign
            print("Res changed to", res)
            old_k = new_k
        else:
            res_step = res_step / 2
            print("Res changed to", res)
        if run > res_epochs:
            print("Exact resolution not found")
            print("Recommended res = ", str(res))
            return res
        run += 1
    print("Recommended res = ", str(res))
    return res


def _compute_CHAOS(clusterlabel, location):

    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)

    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel == k, :]
        if len(location_cluster) <= 2:
            continue
        n_location_cluster = len(location_cluster)
        results = [fx_1NN(i, location_cluster) for i in range(n_location_cluster)]
        dist_val[count] = np.sum(results)
        count = count + 1

    return np.sum(dist_val) / len(clusterlabel)


def fx_1NN(i, location_in):
    location_in = np.array(location_in)
    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    return np.min(dist_array)


def fx_kNN(i, location_in, k, cluster_in):

    location_in = np.array(location_in)
    cluster_in = np.array(cluster_in)

    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    ind = np.argsort(dist_array)[:k]
    cluster_use = np.array(cluster_in)
    if np.sum(cluster_use[ind] != cluster_in[i]) > (k / 2):
        return 1
    else:
        return 0


def _compute_PAS(clusterlabel, location):

    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = location
    results = [
        fx_kNN(i, matched_location, k=10, cluster_in=clusterlabel)
        for i in range(matched_location.shape[0])
    ]
    return np.sum(results) / len(clusterlabel)


def markerFC(adata_valid: AnnData, marker_list, sdm_key):
    rst_dict = {}
    sdm_unique = adata_valid.obs[sdm_key].cat.categories
    for marker in marker_list:
        mean_exp_list = []
        for sdm in sdm_unique:
            mean_exp_list.append(np.mean(adata_valid[adata_valid.obs[sdm_key] == sdm][:, marker].X))
        max_sdm_idx = np.argmax(mean_exp_list)
        #         print(sdm_unique[max_sdm_idx])

        max_sdm_value = np.max(mean_exp_list)
        other_sdm_value = np.mean(
            adata_valid[adata_valid.obs[sdm_key] != sdm_unique[max_sdm_idx]][:, marker].X
        )
        cur_fc = max_sdm_value / other_sdm_value
        rst_dict[marker] = cur_fc
    return rst_dict


def compute_ARI(adata: AnnData, gt_key: str, pred_key: str):
    return adjusted_rand_score(adata.obs[gt_key], adata.obs[pred_key])


def compute_NMI(adata: AnnData, gt_key: str, pred_key: str):
    return normalized_mutual_info_score(adata.obs[gt_key], adata.obs[pred_key])


def compute_CHAOS(adata: AnnData, pred_key: str, spatial_key="spatial"):
    return _compute_CHAOS(adata.obs[pred_key], adata.obsm[spatial_key])


def compute_PAS(adata: AnnData, pred_key: str, spatial_key="spatial"):
    return _compute_PAS(adata.obs[pred_key], adata.obsm[spatial_key])


def compute_ASW(adata: AnnData, pred_key: str, spatial_key="spatial"):
    d = squareform(pdist(adata.obsm[spatial_key]))
    return silhouette_score(X=d, labels=adata.obs[pred_key], metric="precomputed")


def compute_HOM(adata: AnnData, gt_key: str, pred_key: str):
    return homogeneity_score(adata.obs[gt_key], adata.obs[pred_key])


def compute_COM(adata: AnnData, gt_key: str, pred_key: str):
    return completeness_score(adata.obs[gt_key], adata.obs[pred_key])


def marker_score(adata: AnnData, domain_key: str, top_n=5):
    import squidpy as sq

    adata = adata.copy()
    count_dict = adata.obs[domain_key].value_counts()
    adata = adata[adata.obs[domain_key].isin(count_dict.keys()[count_dict > 3].values)]
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.tl.rank_genes_groups(adata, groupby=domain_key)
    selected_genes = []
    for i in range(top_n):
        toadd = list(adata.uns["rank_genes_groups"]["names"][i])
        selected_genes.extend(toadd)
    selected_genes = np.unique(selected_genes)
    sq.gr.spatial_neighbors(adata)
    sq.gr.spatial_autocorr(
        adata,
        mode="moran",
        genes=selected_genes,
        n_perms=100,
        n_jobs=1,
    )
    sq.gr.spatial_autocorr(
        adata,
        mode="geary",
        genes=selected_genes,
        n_perms=100,
        n_jobs=1,
    )
    moranI = np.median(adata.uns["moranI"]["I"])
    gearyC = np.median(adata.uns["gearyC"]["C"])
    return moranI, gearyC


# We use rpy2 to run R packages from Python.
def LISI(coords, meta, label, perplexity=30, nn_eps=0):
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()
    from rpy2.robjects.packages import importr

    importr("lisi")
    if not isinstance(coords, pd.DataFrame):
        coords = pd.DataFrame(coords)
    if not isinstance(meta, pd.DataFrame):
        meta = pd.DataFrame(meta)
    meta = meta.loc[:, [label]]
    meta[label] = meta[label].astype(str)

    coords = robjects.conversion.py2rpy(coords)
    meta = robjects.conversion.py2rpy(meta)
    as_matrix = robjects.r["as.matrix"]
    lisi = robjects.r["compute_lisi"](as_matrix(coords), meta, label, perplexity, nn_eps)
    if isinstance(lisi, pd.DataFrame):
        lisi = lisi.values
    elif isinstance(lisi, np.recarray):
        lisi = [item[0] for item in lisi]

    return lisi
