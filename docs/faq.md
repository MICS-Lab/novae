# Frequently asked questions

### How to use the GPU?

Using a GPU may significantly speed up Novae's training or inference.

If you have a valid GPU for PyTorch, you can set the `accelerator` argument (e.g., one of `["cpu", "gpu", "tpu", "hpu", "mps", "auto"]`) in the following methods: [model.fit()](../api/Novae/#novae.Novae.fit), [model.fine_tune()](../api/Novae/#novae.Novae.fine_tune), [model.compute_representations()](../api/Novae/#novae.Novae.compute_representations).

When using a GPU, we also highly recommend setting multiple workers to speed up the dataset `__getitem__`. For that, you'll need to set the `num_workers` argument in the previous methods, according to the number of CPUs available (`num_workers=8` is usually a good value).

For more details, refer to the API of the [PyTorch Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api) and to the API of the [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

### How to load a pretrained model?

We highly recommend loading a pre-trained Novae model instead of re-training from scratch. For that, choose an available Novae model name on [our HuggingFace collection](https://huggingface.co/collections/MICS-Lab/novae-669cdf1754729d168a69f6bd), and provide this name to the [model.save_pretrained()](../api/Novae/#novae.Novae.save_pretrained) method:

```python
from novae import Novae

model = Novae.from_pretrained("MICS-Lab/novae-human-0") # or any valid model name
```

### How to avoid overcorrecting?

By default, Novae corrects the batch-effect to get shared spatial domains across slides.
The batch information is used only during training (`fit` or `fine_tune`), which should prevent Novae from overcorrecting in `zero_shot` mode.

If not using the `zero_shot` mode, you can provide the `min_prototypes_ratio` parameter to control batch effect correction: either (i) in the `fine_tune` method itself, or (ii) during the model initialization (if retraining a model from scratch).

For instance, if `min_prototypes_ratio=0.5`, Novae expects each slide to contain at least 50% of the prototypes (each prototype can be interpreted as an "elementary spatial domain"). Therefore, the lower `min_prototypes_ratio`, the lower the batch-effect correction. Conversely, if `min_prototypes_ratio=1`, all prototypes are expected to be found in all slides (this doesn't mean the proportions will be the same overall slides, though).

### How do I save my own model?

If you have trained or fine-tuned your own Novae model, you can save it for later use. For that, use the [model.save_pretrained()](../api/Novae/#novae.Novae.save_pretrained) method as below:

```python
model.save_pretrained(save_directory="./my-model-directory")
```

Then, you can load this model back via the [model.from_pretrained()](../api/Novae/#novae.Novae.from_pretrained) method:

```python
from novae import Novae

model = Novae.from_pretrained("./my-model-directory")
```

### How to turn lazy loading on or off?

By default, lazy loading is used only on large datasets. To enforce a specific behavior, you can do the following:

```python
# never use lazy loading
novae.settings.disable_lazy_loading()

# always use lazy loading
novae.settings.enable_lazy_loading()

# use lazy loading only for AnnData objects with 1M+ cells
novae.settings.enable_lazy_loading(n_obs_threshold=1_000_000)
```

### How to update the logging level?

The logging level can be updated as below:

```python
import logging
from novae import log

log.setLevel(logging.ERROR) # or any other level, e.g. logging.DEBUG
```

### How to disable auto-preprocessing?

By default, Novae will preprocess your `adata` object if you provide raw counts. More specifically, it will consider you have raw counts if `adata.X.max() > 10`, and then it will run `sc.pp.normalize_total` and `sc.pp.log1p`. To avoid that, you can run:

```python
novae.settings.auto_preprocessing = False
```

### How to disable the multimodal mode?
By default, Novae will use multimodal mode **if and only if** you ran `novae.compute_histo_embeddings` and `novae.compute_histo_pca`.

To force not using the multimodal mode although you already computed H&E embeddings per cell, you can run:
```python
novae.settings.disable_multimodal = True
```

### How long does it take to use Novae?

The `pip` installation of Novae usually takes less than a minute on a standard laptop. The inference time depends on the number of cells, but typically takes 5-20 minutes on a CPUs, or 30sec to 2 minutes on a GPU (expect it to be roughly 10x times faster on a GPU).

### How to train a new Novae model?

You can decide to train your own Novae model. To do that, simply create a [new model](../api/Novae/#novae.Novae.__init__) as below and use the [`model.fit`](../api/Novae/#novae.Novae.fit) method.

```python
model = novae.Novae(adata) # one or a list of adata objects

model.fit(adata, accelerator="cuda", num_workers=4)
```

### How to monitor the model training?

Since we use [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) to train Novae, you can provide an existing [Lightning Logger](https://lightning.ai/docs/pytorch/stable/extensions/logging.html) to the [`fit`](../api/Novae/#novae.Novae.fit) or [`fine_tune`](../api/Novae/#novae.Novae.fine_tune) methods. For instance, there is an existing logger for [Weight and Biases](https://wandb.ai/site/), which will allow you to aggregate plenty of training metrics, such as loss curves, hardware usage, time, and many others.

If you don't want to use a monitoring library but just want to see the model loss curve, you can also use a CSVLogger and then use our [novae.plot.loss_curve](../api/plot/#novae.plot.loss_curve) function. For instance:

```python
from lightning.pytorch.loggers import CSVLogger

# save the logs in a directory called "logs"
model.fine_tune(logger=CSVLogger("logs"), log_every_n_steps=10)

novae.plot.loss_curve("logs")
```

### How to contribute?

If you want to contribute, check our [contributing guide](https://github.com/MICS-Lab/novae/blob/main/CONTRIBUTING.md).

### How to resolve any other issue?

If you have any bugs/questions/suggestions, don't hesitate to [open a new issue](https://github.com/MICS-Lab/novae/issues).
