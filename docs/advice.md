# Usage advice

Here, we list some advice to help you get the best out of Novae.

### High-quality training subset
When you have many slides, it's recommended to train or fine-tune Novae only on the high-quality slides and run inference (i.e., spatial domain assignment) on all slides. This allows noise removal in the model training while still applying Novae to the whole dataset.

### Resolution or level
When running [`assign_domains`](../api/Novae/#novae.Novae.assign_domains) in **zero-shot**, it may be better to use the `resolution` argument. When fine-tuning or re-training a Novae model, using `level` is recommended.

!!! info
    An advantage of using `level` is that the domains will be nested through the different levels — we don't have such a property using the `resolution` argument.

### Rare tissues
If you have a rare tissue or a tissue that was not used in our large dataset, you might consider re-training a model from scratch. The pre-trained models may work, but if you have low-quality results, it may be interesting to consider re-training a model (see [this tutorial](../tutorials/he_usage/) — you can skip the H&E-embedding section if you don't have an H&E slide aligned).

### Hyperparameters
We recommend using the default Novae hyperparameters, which should work great in most cases. Yet, if you confortable with Novae you might consider updating them. In that case, here are some of the most important hyperparameters in [`fit`](../api/Novae/#novae.Novae.fit) or [`fine_tune`](../api/Novae/#novae.Novae.fine_tune):

- `lr`, the learning rate: you can decrease it, but we recommend values in `[0.0001, 0.001]`.
- `max_epochs`: you can increase it to push the model learning longer. If the model stops because of early stopping, you can also decrease `min_delta` or increase the `patience`.

If you train a new model, you can also change `n_hops_local` and `n_hops_view` (for instance, use 1 for Visium data), a different temperature (around `0.1`), or even make the model bigger - see [here](../api/Novae/#novae.Novae.__init__) the initialization parameters.

If you want to search for the best hyperparameters, we recommend using a monitoring library, see [this FAQ section](../faq/#how-to-monitor-the-model-training).

### Saving a model
If you are satisfied with an existing Novae model that you trained or fine-tuned, you can save it for later usage; see [this FAQ section](../faq/#how-do-i-save-my-own-model).
