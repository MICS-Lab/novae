## Installation

Novae can be installed on every OS via `pip` or [`poetry`](https://python-poetry.org/docs/), on any Python version from `3.9` to `3.12` (included). By default, we recommend using `python==3.10`.

!!! note "Advice (optional)"

    We advise creating a new environment via a package manager, except if you use Poetry, which will automatically create the environment.

    For instance, you can create a new `conda` environment:

    ```bash
    conda create --name novae python=3.10
    conda activate novae
    ```

Choose one of the following, depending on your needs. It should take at most a few minutes.

=== "From PyPI"

    ``` bash
    pip install novae
    ```

=== "pip (editable mode)"

    ``` bash
    git clone https://github.com/MICS-Lab/novae.git
    cd novae

    pip install -e . # no extra
    pip install -e '.[dev]' # all extras
    ```

=== "Poetry (editable mode)"

    ``` bash
    git clone https://github.com/MICS-Lab/novae.git
    cd novae

    poetry install --all-extras
    ```

## Next steps

- We recommend to start with our [first tutorial](../tutorials/main_usage).
- You can also read the [API](../api/novae.Novae).
- If you have questions, please check our [FAQ](../faq) or open an issue on the [GitHub repository](https://github.com/MICS-Lab/novae).
- If you want to contribute, check our [contributing guide](https://github.com/MICS-Lab/novae/blob/main/CONTRIBUTING.md).
