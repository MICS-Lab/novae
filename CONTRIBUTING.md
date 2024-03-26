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

- Use the `black` formatter and `isort`. Their usage should be automatic as they are in the `pyproject.toml` file. Depending on your IDE, you can choose to format your code on save.
- Run `flake8` inside the whole `novae` directory, i.e. `flake8 novae`
- Follow the [PEP8](https://peps.python.org/pep-0008/) style guide.
- Provide meaningful names to all your variables and functions.
- Document your functions and type your function inputs/outputs.
- Try as much as possible to follow the same coding style as the rest of the repository.

## Pull Requests

To add some new code to **novae**, you should:

1. Fork the repository
2. Install `novae` in editable mode with the 'dev' extra (see above)
3. Create your personal branch from `main`
4. Implement your changes
5. Create a pull request on the `main` branch. Add explanations about your developed features, and wait for discussion and validation of your pull request
