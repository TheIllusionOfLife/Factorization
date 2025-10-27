# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM-guided evolutionary algorithm that optimizes heuristics for the General Number Field Sieve (GNFS) factorization algorithm. The system evolves strategies that identify "smooth numbers" (numbers with many small prime factors) using either rule-based mutations or Gemini LLM-guided mutations.

## Development Commands

### Running the Application

**Rule-based mode** (no LLM):
```bash
python prototype.py --generations 5 --population 10
```

**LLM-guided mode** (requires `GEMINI_API_KEY` in `.env`):
```bash
python prototype.py --llm --generations 5 --population 10
```

**Common CLI options**:
- `--number NUMBER`: Target number to factor (default: 961730063)
- `--generations N`: Number of evolutionary generations (default: 5)
- `--population N`: Population size per generation (default: 10)
- `--duration SECS`: Evaluation duration in seconds (default: 0.1)
- `--llm`: Enable LLM-guided mutations using Gemini 2.5 Flash Lite
- `--export-metrics PATH`: Export detailed metrics to JSON file for analysis

### Testing

**Run all tests**:
```bash
pytest tests/ -v
```

**Run specific test file**:
```bash
pytest tests/test_integration.py -v
```

**Run real API integration test** (requires `GEMINI_API_KEY`):
```bash
GEMINI_API_KEY=your_key pytest tests/test_llm_provider.py::test_real_gemini_call -v
```

### Continuous Integration

**CI Workflow** runs automatically on PRs:
```bash
# What CI checks
- pytest: All unit and integration tests (Python 3.9, 3.10, 3.11)
- ruff check: Linting (replaces Flake8, isort, pyupgrade)
- ruff format --check: Code formatting (replaces Black)
- mypy: Type checking (optional, may have false positives)
```

**Local CI simulation**:
```bash
# Run all checks locally before pushing
pytest tests/ -v
ruff check .
ruff format --check .
mypy src/ --ignore-missing-imports
```

**Auto-fix issues**:
```bash
# Fix linting and formatting issues automatically
ruff check . --fix
ruff format .
```

### Setup

**Install dependencies**:
```bash
pip install -r requirements.txt
```

**Configure environment**:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Architecture

### Core Components

**prototype.py** - Main evolutionary engine with three key classes:

1. **FactorizationCrucible**: The evaluation environment
   - Simulates GNFS sieving step
   - Evaluates strategies by counting "smooth" candidates found in fixed time
   - Uses target number N and search space around sqrt(N)

2. **EvolutionaryEngine**: Orchestrates the evolutionary process
   - Manages population of strategies across generations
   - Selection: Top 20% become parents (elite selection)
   - Mutation: 80% mutated from parents, 20% random newcomers
   - Uses LLMStrategyGenerator if LLM mode enabled, else basic StrategyGenerator

3. **Strategy**: Dataclass representing optimization heuristics
   - `power` (2-5): Polynomial degree for candidate generation (x^power - N)
   - `modulus_filters`: List of (modulus, residues) for fast rejection
   - `smoothness_bound`: Maximum prime factor to check (from SMALL_PRIMES)
   - `min_small_prime_hits`: Required count of small prime factors

4. **EvaluationMetrics**: Detailed metrics tracking (NEW in PR #8)
   - `candidate_count`: Total smooth candidates found
   - `smoothness_scores`: Quality metrics (lower = smoother)
   - `timing_breakdown`: Time spent in each evaluation phase
   - `rejection_stats`: Counts of rejection reasons
   - `example_candidates`: Sample smooth numbers found

### Metrics & Instrumentation

**evaluate_strategy_detailed()**: Comprehensive evaluation method
- Tracks timing using `time.perf_counter()` for high precision
- Separates timing into three phases:
  - Candidate generation: `pow(x, power) - N`
  - Modulus filtering: Fast rejection via residue checks
  - Smoothness check: Prime factorization counting
- Calculates smoothness ratio: `candidate / product_of_small_primes`
- Limits stored data: 10 smoothness scores, 5 example candidates

**metrics_history**: List[List[EvaluationMetrics]]
- Nested structure: `[generation][civilization]`
- Stored in EvolutionaryEngine for entire evolution
- Exported to JSON via `export_metrics(output_path)`

**Visualization**: Jupyter notebook (analysis/visualize_metrics.ipynb)
- Fitness trends over generations
- Timing breakdown analysis (stacked area charts)
- Rejection statistics (bar charts)
- Smoothness quality evolution
- Best strategy analysis with detailed stats

### LLM Integration (src/llm/)

**base.py**: Abstract interfaces
- `LLMProvider`: Abstract base class for LLM providers
- `LLMResponse`: Standardized response dataclass with success, mutation_params, reasoning, cost tracking

**gemini.py**: Gemini API implementation
- Uses Gemini 2.5 Flash Lite with structured JSON output (`response_schema`)
- Temperature scaling: High early (exploration) → Low late (exploitation)
  - Formula: `temperature_max - (temperature_max - temperature_base) * progress`
  - Prevents inverted exploration-exploitation tradeoff
- Cost tracking with defensive token access (handles API changes gracefully)
- **Critical**: Increments call counter in `finally` block to prevent infinite retries on errors

**schemas.py**: Pydantic response schemas
- `MutationResponse`: Top-level schema with mutation_type and corresponding params
- Five mutation types: power, add_filter, modify_filter, remove_filter, adjust_smoothness
- Field validation: Ensures residues < modulus, power in [2,5], etc.

### Configuration (src/config.py)

**Config dataclass**:
- `api_key`: Gemini API key from environment
- `max_llm_calls`: Limit on API calls per run (default: 100)
- `temperature_base`: Starting temperature for exploitation (default: 0.8)
- `temperature_max`: Starting temperature for exploration (default: 1.2)
- `temperature_scaling_generations`: Generations to scale from max→base (default: 10)

**Validation**: Raises ValueError if LLM enabled but API key missing (fail-fast pattern)

## Critical Implementation Details

### Temperature Calculation Bug (PR #2)

**Issue**: Original implementation increased temperature (0.8→1.2) over generations instead of decreasing
**Impact**: Inverted exploration-exploitation tradeoff fundamental to evolutionary algorithms
**Fix**: Correct formula starts at `temperature_max` and decreases to `temperature_base`

### API Call Counting Pattern

**Requirement**: Count ALL API calls (success or failure) to prevent resource leaks
**Implementation**: Use `finally` block in gemini.py:propose_mutation()
```python
try:
    # API call
except Exception:
    # Error handling
finally:
    self._call_count += 1  # ALWAYS increment
```

### Strategy Normalization

**Purpose**: Ensures all strategies have valid parameters after mutation
**Location**: Strategy._normalize() called in __post_init__
**Rules**:
- Power: Clamp to [2, 5]
- Modulus filters: Max 4 filters, residues < modulus
- Smoothness bound: Must be in SMALL_PRIMES list
- Min hits: Clamp to [1, 6]

### LLM Mutation Application

**Pattern**: LLMStrategyGenerator._apply_llm_mutation() handles 5 mutation types
**Defensive coding**:
- Validates filter indices before access
- Logs warnings but continues with parent strategy if invalid
- Ensures min 1 filter (cannot remove last filter)
- Max 4 filters (cannot exceed limit)

### Fitness History Management

**Issue**: History list grew unbounded (memory leak)
**Fix**: Keep only last 5 entries in EvolutionaryEngine.run_evolutionary_cycle():
```python
if len(self.generator.fitness_history) > 5:
    self.generator.fitness_history = self.generator.fitness_history[-5:]
```

## Testing Patterns

### Integration Tests (tests/test_integration.py)

- Test full evolutionary cycles with mock LLM
- Verify strategy normalization and mutation logic
- Test both LLM and rule-based modes

### LLM Provider Tests (tests/test_llm_provider.py)

- Mock Gemini API responses with structured JSON
- Test error handling (JSONDecodeError, ValueError, API errors)
- Verify token tracking and cost calculation
- Real API test marked separately (requires actual API key)

### Schema Validation Tests (tests/test_schemas.py)

- Test Pydantic field validators
- Verify residues < modulus constraint
- Test all five mutation types with valid/invalid inputs

## Known Issues & Next Steps

### From PR #2 Review (Low Priority)

1. **CI/CD Organization**: 5 Gemini workflows (1,250+ lines) could be better organized
2. **Production Logging**: Logger created but not configured with appropriate log levels
3. **Magic Numbers**: Hardcoded 0.2 for elite selection, 0.8/1.2 temperatures should move to config

### Limitations

- This is a **prototype/research tool**, not production factorization software
- Uses simplified GNFS sieving heuristics
- Evaluates "smoothness" not actual factorization
- For production factorization, use CADO-NFS or msieve

## Cost Estimates

Gemini 2.5 Flash Lite pricing:
- Free tier: Unlimited requests within rate limits
- Paid tier: $0.10/M input tokens, $0.40/M output tokens

Typical prototype costs:
- 3 generations, 5 population: ~$0.001-0.002 (13 API calls)
- 10 generations, 10 population: ~$0.005-0.010 (40-50 API calls)

## Development Workflow

1. **Create feature branch** before any changes
2. **Write tests first** for new features (TDD)
3. **Run tests locally** before committing
4. **Commit at logical milestones** with conventional commit format
5. **Push to GitHub** and create PR (never push to main)
6. **Verify CI passes** before requesting review

## Critical Learnings from PR #2

1. **Temperature Inversion**: Always verify exploration→exploitation patterns in evolutionary algorithms
2. **Finally Block Pattern**: Use `finally` for counters to prevent resource leaks on failures
3. **Early Validation**: Validate config (API keys) at load time with clear error messages
4. **Dataclass Equality**: Python's `@dataclass` auto-generates `__eq__` (no manual implementation needed)
5. **Specific Exception Handling**: Catch specific exceptions (JSONDecodeError, ValueError) not bare Exception
