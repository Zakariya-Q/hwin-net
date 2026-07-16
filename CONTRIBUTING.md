# Contributing to HWIN-Net

Thank you for your interest in contributing! HWIN-Net is a research implementation
of a theory-derived neural architecture for water quality prediction.

## Getting Started

1. Fork the repository
2. Clone your fork: git clone https://github.com/YOUR-USERNAME/hwin-net.git
3. Install in development mode: pip install -e .[dev]
4. Run tests: pytest

## Development Workflow

### Code Style

- Format: lack (line length 100)
- Lint: uff (see pyproject.toml)
- Types: mypy (see pyproject.toml)

Run all checks:
`ash
black .
ruff check .
mypy hwin_net
`

### Testing

`ash
# Run all tests
pytest

# Run with coverage
pytest --cov=hwin_net

# Run specific test file
pytest tests/test_theory.py -v
`

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- eat: - New feature
- ix: - Bug fix
- docs: - Documentation changes
- efactor: - Code restructuring
- 	est: - Test additions/changes
- ci: - CI configuration

Example: eat(models): add transformer encoder option

## Pull Request Process

1. Create a feature branch from main
2. Make changes with tests
3. Run full test suite: pytest
4. Run linting: lack . && ruff check . && mypy hwin_net
5. Update documentation if needed
6. Submit PR with clear description

### PR Requirements

- All tests pass
- No linting errors
- No new type errors
- Documentation updated for user-facing changes
- CHANGELOG.md updated (if applicable)

## Scientific Code

**The scientific implementation is FROZEN.** 

Do not modify:
- Mathematical theory (axioms, lemmas, theorems)
- Architecture modules (M1-M6)
- Loss functions
- Benchmark protocol
- Hyperparameters in configs

Only modify:
- Packaging, imports, repository structure
- Documentation
- Build system, CI, deployment
- Tests (engineering fixes only)

## Reporting Issues

Use GitHub Issues for:
- Bug reports (engineering only)
- Documentation improvements
- Build/installation issues
- CI failures

Do NOT use issues for:
- Scientific methodology changes
- Architecture modifications
- Hyperparameter tuning requests

## Code of Conduct

All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
