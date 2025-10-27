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
- **`bot-gemini-dispatch.yml`**: Main dispatcher for Gemini bot commands
- **`bot-gemini-review.yml`**: Gemini PR review workflow (reusable)
- **`bot-gemini-invoke.yml`**: Gemini invocation workflow (reusable)
- **`bot-gemini-triage.yml`**: Gemini issue triage workflow (reusable)
- **`bot-gemini-scheduled.yml`**: Scheduled Gemini triage

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

### Gemini Dispatch (`bot-gemini-dispatch.yml`)
- **Pull Requests**: When opened (skips docs-only PRs)
- **Issues**: When opened or reopened
- **Comments**: When @gemini-cli is mentioned by OWNER/MEMBER/COLLABORATOR

### Claude Workflows
- **`bot-claude.yml`**: Issue/PR comments mentioning @claude
- **`bot-claude-review.yml`**: Pull request opened events

## Path Filters

### Gemini Review Skips
Documentation-only PRs skip Gemini review to avoid unnecessary failures:
- `**.md` files
- `**.txt` files
- `docs/**` directory
- `LICENSE`, `.gitignore`, `.env.example`

### Python CI Triggers
Only runs when relevant files change:
- `**.py` files
- `requirements.txt`
- `pyproject.toml`

## Troubleshooting

### "Workflow file not found" errors
- Ensure reusable workflow paths in `bot-gemini-dispatch.yml` are correct
- Reusable workflows must be in `.github/workflows/` directory

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
