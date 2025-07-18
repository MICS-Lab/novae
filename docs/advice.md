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

### Saving a model
If you are satisfied with an existing Novae model that you trained or fine-tuned, you can save it for later usage; see [this FAQ section](../faq/#how-do-i-save-my-own-model).
