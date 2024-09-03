import seaborn as sns
from anndata import AnnData


def get_categorical_color_palette(adatas: list[AnnData], obs_key: str) -> tuple[list[str], list[str]]:
    key_added = f"{obs_key}_colors"

    all_domains = sorted(list(set.union(*[set(adata.obs[obs_key].cat.categories) for adata in adatas])))

    n_colors = len(all_domains)
    colors = list(sns.color_palette("tab10" if n_colors <= 10 else "tab20", n_colors=n_colors).as_hex())
    for adata in adatas:
        adata.obs[obs_key] = adata.obs[obs_key].cat.set_categories(all_domains)
        adata.uns[key_added] = colors

    return all_domains, colors
