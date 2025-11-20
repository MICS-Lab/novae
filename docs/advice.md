# Usage advice

Here, we list some advice to help you get the best out of Novae.

### High-quality training subset
When you have many slides, it's recommended to train or fine-tune Novae only on the good quality slides and run inference (i.e., spatial domain assignment) on all slides. This allows noise removal in the model training while still applying Novae to the low quality slides.

### Resolution or level
When running [`assign_domains`](../api/Novae/#novae.Novae.assign_domains) in **zero-shot**, it may be better to use the `resolution` argument. When fine-tuning or re-training a Novae model, using `level` is recommended.

!!! info
    An advantage of using `level` is that the domains will be nested through the different levels — we don't have such a property using the `resolution` argument.

### Coordinate system
After computing the Delaunay graph, Novae uses the distances between cells for training and during inference. This can be useful as it helps to better separate spatial domains with different densities of cells.
By default, we expect `adata.obsm["spatial"]` coordinates to be in microns.

If your coordinates are **not** in microns, set `novae.settings.scale_to_microns` to the conversion factor from your coordinate units to microns. For instance, if you have Xenium data and use a pixel coordinate system, you can set `novae.settings.scale_to_microns = 0.2125`.

!!! info
    Using `novae.settings.scale_to_microns` will not affect the Delaunay graph itself, it will only be used by Novae during training/inference.

### Rare tissues
If you have a rare tissue or a tissue that was not used in our large dataset, you might consider re-training a model from scratch. The pre-trained models may work, so **try them first**, but if you have low-quality results, then it may be interesting to consider re-training a model. To do that, see [this tutorial](../tutorials/he_usage/) — you can skip the H&E-embedding section if you don't have an H&E slide aligned.

### Using references
For the zero-shot and fine-tuning modes, you can provide a `reference` slide (or multiple slides). This allows to recompute the model prototypes (i.e., the centroids of the spatial domains) based on the chosen slides.

- For [zero-shot](../api/Novae/#novae.Novae.compute_representations), we use `reference="all"` by default, meaning we use all slides to recompute the prototypes. Depending on your use case, you may consider specifying one or multiple **representative** slides.
- For [fine-tuning](../api/Novae/#novae.Novae.fine_tune), we use `reference=None` by default, meaning we will initialize the prototypes randomly, and re-train them. **If you have only one slide**, it may be worth trying `reference="all"`.

### Handling large datasets

Novae uses lazy-loading for the model training (i.e., you don't need a lot of GPU memory), but you still need to be able to load your dataset on CPUs. We recommend using sparse `csr_matrix` in `adata.X` by default, but, if your dataset becomes too large, sparse matrices may not be enough anymore.

In that case, you can use other backends (two options shown below).

=== "Backed H5"

    You can read your data in "backed mode" to keep `adata.X` on disk and load only the requested slices into memory when needed. It enables scalable operations on large datasets without fully materializing them in RAM.

    For instance, you can read an `h5ad` file as below:
    ```python
    adata = anndata.read_h5ad("large_data.h5ad", backed='r')
    ```

=== "Dask"

    You can use Dask array to lazy load `adata.X` (see the [AnnData tutorials](https://anndata.readthedocs.io/en/stable/tutorials/index.html) to use Dask). You don't need to change anything in your code, `novae` will handle the Dask backend for you!

    !!! info "Chunk sizes"
        The chunk size will influence how fast the mini-batches are created. We will soon perform some benchmarks to see how best to choose the chunk size.

### Hyperparameters
We recommend using the default Novae hyperparameters, which should work great in most cases. Yet, if you confortable with Novae you might consider updating them. In that case, here are some of the most important hyperparameters in [`fit`](../api/Novae/#novae.Novae.fit) or [`fine_tune`](../api/Novae/#novae.Novae.fine_tune):

- `lr`, the learning rate: you can decrease it, but we recommend values in `[0.0001, 0.001]`.
- `max_epochs`: you can increase it to push the model learning longer. If the model stops because of early stopping, you can also decrease `min_delta` or increase the `patience`.

If you train a new model, you can also change `n_hops_local` and `n_hops_view` (for instance, use 1 for Visium data), a different temperature (around `0.1`), or even make the model bigger - see [here](../api/Novae/#novae.Novae.__init__) the initialization parameters.

If you want to search for the best hyperparameters, we recommend using a monitoring library, see [this FAQ section](../faq/#how-to-monitor-the-model-training).

### Saving a model
If you are satisfied with an existing Novae model that you trained or fine-tuned, you can save it for later usage; see [this FAQ section](../faq/#how-do-i-save-my-own-model).
