# GitHub Actions Workflows

This directory contains all CI/CD workflows for the Factorization project.

## Workflow Organization

### Python CI
- **`ci-python.yml`**: Main CI pipeline for Python code
  - Runs pytest on Python 3.9, 3.10, 3.11
  - Linting with Ruff
  - Type checking with mypy
  - Coverage reporting to Codecov

### AI Bot Integrations
- **`bot-claude.yml`**: Claude Code bot integration
- **`bot-claude-review.yml`**: Automated Claude PR reviews

**Note**: Code reviews are also automatically provided by the [Gemini Code Assist GitHub App](https://github.com/marketplace/gemini-code-assist).

## Local Development

### Running Tests Locally
```bash
pytest tests/ -v
```

### Running Linter
```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check . --fix

# Format code
ruff format .
```

### Running Type Checker
```bash
mypy src/ --ignore-missing-imports
```

## Workflow Triggers

### Python CI (`ci-python.yml`)
- **Pull Requests**: When Python files, requirements.txt, or pyproject.toml change
- **Push to main**: Same path filters

### Claude Workflows
- **`bot-claude.yml`**: Issue/PR comments mentioning @claude
- **`bot-claude-review.yml`**: Pull request opened events

## Path Filters

### Python CI Triggers
Only runs when relevant files change:
- `**.py` files
- `requirements.txt`
- `pyproject.toml`

## Troubleshooting

### Ruff failures
```bash
# Run locally to see exact issues
ruff check . --output-format=full

# Auto-fix most issues
ruff check . --fix
```

### Test failures
```bash
# Run specific test file
pytest tests/test_integration.py -v

# Run with detailed output
pytest tests/ -vv --tb=long
```

## Adding New Workflows

1. Create workflow file in `.github/workflows/`
2. Use descriptive naming convention:
   - `ci-*` for CI/CD pipelines
   - `bot-*` for AI bot integrations
   - `deploy-*` for deployment workflows
3. Add path filters to avoid unnecessary runs
4. Document trigger conditions in this README
5. Test on feature branch before merging

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [pytest Documentation](https://docs.pytest.org/)
