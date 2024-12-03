# Frequently asked questions

### How to use the GPU?

Using a GPU may significantly speed-up the training or inference of Novae.

If you have a valid GPU for PyTorch, you can set the `accelerator` argument (e.g., one of `["cpu", "gpu", "tpu", "hpu", "mps", "auto"]`) in the following methods: [model.fit()](../api/Novae/#novae.Novae.fit), [model.fine_tune()](../api/Novae/#novae.Novae.fine_tune), [model.compute_representations()](../api/Novae/#novae.Novae.compute_representations).

When using a GPU, we also highly recommend setting multiple workers to speed up the dataset `__getitem__`. For that, you'll need to set the `num_workers` argument in the previous methods, according to the number of CPUs available (`num_workers=8` is usually a good value).

For more details, refer to the API of the [PyTorch Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api) and to the API of the [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

### How to load a pretrained model?

We highly recommend to load a pretrained Novae model instead of re-training from scratch. For that, choose an available Novae model name on [our HuggingFace collection](https://huggingface.co/collections/MICS-Lab/novae-669cdf1754729d168a69f6bd), and provide this name to the [model.save_pretrained()](../api/Novae/#novae.Novae.save_pretrained) method:

```python
from novae import Novae

model = Novae.from_pretrained("MICS-Lab/novae-human-0") # or any valid model name
```


### How to save my own model?

If you have trained or fine-tuned your own Novae model, you can save it for later use. For that, use the [model.save_pretrained()](../api/Novae/#novae.Novae.save_pretrained) method as below:

```python
model.save_pretrained(save_directory="./my-model-directory")
```

Then, you can load this model back via the [model.from_pretrained()](../api/Novae/#novae.Novae.from_pretrained) method:

```python
from novae import Novae

model = Novae.from_pretrained("./my-model-directory")
```

### How to disable or enable lazy loading?

By default, lazy loading is used only on large datasets. To enforce a specific behaviour, you can do the following:

```python
# never use lazy loading
novae.settings.disable_lazy_loading()

# always use lazy loading
novae.settings.enable_lazy_loading()

# use lazy loading only for AnnData objects with 1M+ cells
novae.settings.enable_lazy_loading(n_obs_threshold=1_000_000)
```

### How to update the logging level?

The logging level can be update as below:

```python
import logging
from novae import log

log.setLevel(logging.ERROR) # or any other level, e.g. logging.DEBUG
```

### How to contribute?

If you want to contribute, check our [contributing guide](https://github.com/MICS-Lab/novae/blob/main/CONTRIBUTING.md).

### How to resolve any other issue?

If you have any bug/question/suggestion, don't hesitate to [open a new issue](https://github.com/MICS-Lab/novae/issues).
