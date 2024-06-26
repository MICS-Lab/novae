# Contributing to *novae*

Contributions are welcome as we aim to continue improving `novae`. For instance, you can contribute by:

- Opening an issue
- Discussing the current state of the code
- Making a Pull Request (PR)

If you want to open a PR, follow the following instructions.

## Installing `novae` in editable mode

When contributing, installing `novae` in editable mode is recommended. Also, we recommend installing the 'dev' extra.

For this, use one of the two following lines:

```sh
# with pip
pip install -e '.[dev]'

# or with poetry
poetry install -E dev
```

## Coding guidelines

We use `pre-commit` to run code quality controls before the commits. This includes running `black`, `isort`, `flake8`, and others.


After installing `pre-commit`, you can set it up at the root of `novae` like this:
```sh
pre-commit install
```

Then, it will run the pre-commit automatically before commit. But you can also run the pre-commit manually:
```sh
pre-commit run --all-files
```

Apart from this, we also recommend to follow the standard styling conventions:
- Follow the [PEP8](https://peps.python.org/pep-0008/) style guide.
- Provide meaningful names to all your variables and functions.
- Type your function inputs/outputs, and add docstrings in the Google style.
- Try as much as possible to follow the same coding style as the rest of the repository.

## Pull Requests

To add some new code to **novae**, you should:

1. Fork the repository
2. Install `novae` in editable mode with the 'dev' extra (see above)
3. Create your personal branch from `main`
4. Implement your changes
5. Create a pull request on the `main` branch. Add explanations about your developed features, and wait for discussion and validation of your pull request
