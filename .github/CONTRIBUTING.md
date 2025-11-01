# Contributing to Factorization

Thank you for your interest in contributing! This guide will help you set up your development environment and understand our development workflow.

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/TheIllusionOfLife/Factorization.git
cd Factorization

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Install Pre-Commit Hooks (Required)

**This step is mandatory for all contributors.** Pre-commit hooks automatically validate your code before commits, preventing CI failures.

```bash
make install-hooks
```

This installs hooks that will run automatically on `git commit`:
- **ruff-check**: Code linting
- **ruff-format**: Code formatting
- **pytest-fast**: Fast unit tests
- **trailing-whitespace**: Remove trailing spaces
- **end-of-file-fixer**: Ensure files end with newline
- **check-yaml**: Validate YAML files
- **check-merge-conflict**: Prevent committing merge conflicts

**Performance**: Hooks run in <10 seconds for typical commits.

---

## Development Workflow

### Branch Strategy

1. **Always create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Never commit directly to `main`** - Pre-commit hooks will block this.

### Local CI Validation

Before pushing, run local CI checks to ensure your code will pass GitHub Actions CI:

```bash
# Run all CI checks locally (recommended)
make ci

# Or run fast checks only
make ci-fast
```

### Individual Checks

```bash
# Linting only
make lint

# Auto-fix formatting
make format

# Run all tests
make test

# Run fast tests (exclude integration)
make test-fast

# Type checking
make type-check
```

### What Pre-Commit Hooks Do

When you run `git commit`, hooks automatically:

1. **Check and fix code formatting** (ruff format)
2. **Check and fix linting issues** (ruff check with --fix)
3. **Run fast unit tests** (pytest excluding integration tests)
4. **Remove trailing whitespace**
5. **Ensure files end with newline**
6. **Validate YAML files**

**If any hook fails**, the commit is blocked and you'll see:
```
❌ ruff-check.............Failed
- hook id: ruff
- exit code: 1
```

### Fixing Hook Failures

**1. Read the error message carefully** - it tells you exactly what's wrong.

**2. Fix the issue**:
```bash
# For formatting issues
make format

# For linting issues
make lint

# For test failures
pytest tests/test_specific_file.py -v
```

**3. Try committing again**:
```bash
git add .
git commit -m "your message"
```

### Emergency Bypass (Use Sparingly!)

If you absolutely need to bypass hooks (e.g., WIP commit), use:
```bash
git commit --no-verify -m "WIP: incomplete work"
```

**⚠️ Warning**: CI will still run and may fail. Only use for WIP commits on feature branches.

---

## Test-Driven Development (TDD)

We strongly encourage TDD for new features:

1. **Write tests first** that define expected behavior
2. **Confirm tests fail** (`pytest tests/`)
3. **Write minimal code** to make tests pass
4. **Refactor** once tests are green
5. **Commit tests and implementation separately** for clear history

### Running Tests

```bash
# All tests
make test

# Specific test file
pytest tests/test_strategy.py -v

# Specific test function
pytest tests/test_strategy.py::test_strategy_normalization -v

# With coverage
make test-cov
```

### Test Organization

- **Unit tests**: `tests/test_*.py` - Fast, isolated tests
- **Integration tests**: Marked with `@pytest.mark.integration` - Slower, test component interactions
- **Fixtures**: `tests/conftest.py` - Shared test fixtures

---

## Code Quality Standards

### Python Style

- **Linting**: Ruff (replaces Flake8, isort, pyupgrade, Black)
- **Formatting**: Ruff format (Black-compatible)
- **Line length**: 88 characters
- **Type hints**: Encouraged but not required
- **Docstrings**: Required for public functions/classes

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code changes that neither fix bugs nor add features
- `perf`: Performance improvements
- `style`: Code style changes (formatting, semicolons, etc.)
- `ci`: CI/CD changes

**Examples**:
```
feat: add LLM-guided mutation operator

Implements semantic mutation suggestions using Gemini 2.5 Flash API.
Includes temperature scaling and structured JSON output validation.

Closes #42
```

```
fix: resolve race condition in parallel evaluation

Population evaluations were not thread-safe. Added locks around
fitness history updates.
```

---

## Pull Request Process

1. **Ensure all checks pass**:
   ```bash
   make ci
   ```

2. **Update documentation** if you changed:
   - Public APIs (update docstrings)
   - CLI commands (update README.md, CLAUDE.md)
   - Configuration options (update CLAUDE.md)

3. **Create PR with descriptive title and body**:
   - Explain what changed and why
   - Link related issues
   - Include testing notes

4. **Wait for CI and code review**:
   - All CI checks must pass (tests, linting, type checking)
   - Address review feedback promptly
   - Update PR description if scope changes

5. **Squash merge** is used for clean commit history

---

## Common Issues

### Hook Installation Failed

**Problem**: `make install-hooks` fails with "pre-commit not found"

**Solution**:
```bash
pip install -r requirements-dev.txt
make install-hooks
```

### Hooks Are Too Slow

**Problem**: Hooks take >10 seconds on first commit

**Solution**: First run installs hook environments (one-time setup). Subsequent commits will be fast (<10s).

### Test Failures

**Problem**: `pytest-fast` hook fails

**Solution**:
1. Read error message to identify failing test
2. Run test locally: `pytest tests/test_file.py::test_name -v`
3. Fix the issue
4. Verify: `pytest tests/test_file.py -v`
5. Commit again

### Import Errors

**Problem**: `ModuleNotFoundError` in tests

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

## Additional Resources

- **README.md**: Project overview and usage
- **CLAUDE.md**: Development guidelines and architecture
- **research_methodology.md**: Research methodology and experimental design
- **Makefile**: All available development commands (`make help`)

---

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Tag issues appropriately (bug, enhancement, question, etc.)

Thank you for contributing! 🚀
