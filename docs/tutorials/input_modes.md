Depending on your data and preferences, you case use 4 types of inputs.
Specifically, it depends whether (i) you have one or multiple slides, and (ii) if you prefer to concatenate your data or not.

!!! info
    In all cases, the data structure is [AnnData](https://anndata.readthedocs.io/en/latest/). We may support MuData in the future.

## 1. One slide mode

This case is the easiest one. You simply have one `AnnData` object corresponding to one slide.

You can follow the first section of the [main usage tutorial](../main_usage).

## 2. Multiple slides, one AnnData object

If you have multiple slides with the same gene panel, you can concatenate them into one `AnnData` object. In that case, make sure you keep a column in `adata.obs` that denotes which cell corresponds to which slide.

Then, remind this column, and pass it to the two following Novae methods: [compute_representations](../../api/novae.Novae/#novae.Novae.compute_representations) and [fine_tune](../../api/novae.Novae/#novae.Novae.fine_tune). The other functions don't need this argument.

!!! example
    For instance, you can do:
    ```python
    model.compute_representations(adata, slide_key="my-slide-id-column", zero_shot=True)
    ```

## 3. Multiple slides, one AnnData object per slide

If you have multiple slides, you may prefer to keep one `AnnData` object for each slide. This is also convenient if you have different gene panels and can't concatenate your data.

That case is pretty easy, since most functions and methods of Novae also support a **list of `AnnData` objects** as inputs. Therefore, simply pass a list of `AnnData` object, as below:

!!! example
    ```python
    adatas = [adata_1, adata_2, ...]

    model.compute_representations(adatas, zero_shot=True)
    ```

## 4. Multiple slides, multiple slides per AnnData object

If you have multiple slides and multiple panels, an alternative to option 3 is to have one `AnnData` object per panel, and multiple slides inside each `AnnData` object. In that case, make sure you keep a column in `adata.obs` that denotes which cell corresponds to which slide.

Then, remind this column, and pass it to the two following Novae methods: [compute_representations](../../api/novae.Novae/#novae.Novae.compute_representations) and [fine_tune](../../api/novae.Novae/#novae.Novae.fine_tune). The other functions don't need this argument.

!!! example
    For instance, you can do:
    ```python
    adatas = [adata_1, adata_2, ...]

    model.compute_representations(adatas, slide_key="my-slide-id-column", zero_shot=True)
    ```