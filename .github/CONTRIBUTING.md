# Contributing to Star Trek Technology Project

We love your input! We want to make contributing to Star Trek Technology Project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Pull Request Process

1. Update the README.md with details of changes to the interface
2. Update the docs/ with any necessary changes
3. The PR will be merged once you have the sign-off of two other developers

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/Saifullah62/starTrek_tech/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/Saifullah62/starTrek_tech/issues/new).

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Setup

1. Create a virtual environment:
```bash
make setup-env
```

2. Install dependencies:
```bash
make dev-install
```

3. Run tests:
```bash
make test
```

4. Check code style:
```bash
make lint
```

## Code Style

- We use [Black](https://github.com/psf/black) for Python code formatting
- We use [isort](https://pycqa.github.io/isort/) for import sorting
- We use [mypy](http://mypy-lang.org/) for static type checking
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide

## Documentation

- Write docstrings for all public modules, functions, classes, and methods
- Use [Google style](https://google.github.io/styleguide/pyguide.html) for docstrings
- Keep the [docs/](docs/) folder up to date

## Testing

- Write unit tests for all new code
- Maintain or improve test coverage
- Use pytest for testing
- Place tests in the `tests/` directory

## Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
