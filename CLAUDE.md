# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM-guided evolutionary algorithm that optimizes heuristics for the General Number Field Sieve (GNFS) factorization algorithm. The system evolves strategies that identify "smooth numbers" (numbers with many small prime factors) using either rule-based mutations or Gemini LLM-guided mutations.

## Development Commands

### Running the Application

**Note**: Use `main.py` (new modular entry point) or `prototype.py` (backward compatibility shim).

**Rule-based mode** (no LLM):
```bash
python main.py --generations 5 --population 10
```

**LLM-guided mode** (requires `GEMINI_API_KEY` in `.env`):
```bash
python main.py --llm --generations 5 --population 10
```

**Comparison mode** (statistical analysis against baselines):
```bash
# Rule-based comparison with 5 independent runs
python main.py --compare-baseline --num-comparison-runs 5 --generations 10 --population 10 --seed 42

# LLM-guided comparison with JSON export
python main.py --llm --compare-baseline --num-comparison-runs 3 --generations 5 \
  --export-comparison results/comparison.json --seed 100
```

**Common CLI options**:
- `--number NUMBER`: Target number to factor (default: 961730063)
- `--generations N`: Number of evolutionary generations (default: 5)
- `--population N`: Population size per generation (default: 10)
- `--duration SECS`: Evaluation duration in seconds (default: 0.1)
- `--llm`: Enable LLM-guided mutations using Gemini 2.5 Flash Lite
- `--export-metrics PATH`: Export detailed metrics to JSON file for analysis
- `--compare-baseline`: Run comparison against baseline strategies
- `--num-comparison-runs N`: Number of independent comparison runs (default: 5)
- `--convergence-window N`: Generation window for convergence detection (default: 5)
- `--export-comparison PATH`: Export comparison results to JSON file

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

### Local CI Validation

**Makefile targets** for convenient local validation:
```bash
# Run all CI checks locally (recommended before pushing)
make ci

# Run fast CI checks (lint + format + fast tests)
make ci-fast

# Individual checks
make lint          # Run ruff linter
make format        # Auto-fix formatting issues
make format-check  # Check formatting without modifying
make test          # Run all tests
make test-fast     # Run fast tests only (exclude integration)
make type-check    # Run mypy type checking
```

**CI Workflow** runs automatically on PRs:
```bash
# What CI checks (matches local `make ci`)
- pytest: All unit and integration tests (Python 3.9, 3.10, 3.11)
- ruff check: Linting (replaces Flake8, isort, pyupgrade)
- ruff format --check: Code formatting (replaces Black)
- mypy: Type checking (optional, may have false positives)
```

**Pre-commit hooks** run automatically on `git commit`:
- Hooks match CI checks exactly (except integration tests excluded for speed)
- Block commits that would fail CI
- Run in <10 seconds for typical commits
- Bypass with `git commit --no-verify` (emergency only)

### Setup

**Install dependencies**:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Configure environment**:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

**Install pre-commit hooks** (required for contributors):
```bash
make install-hooks
```

This installs Git hooks that automatically run before each commit:
- **ruff-check**: Linting with auto-fix
- **ruff-format**: Code formatting
- **pytest-fast**: Fast unit tests (excludes integration tests)
- **trailing-whitespace**: Remove trailing spaces
- **end-of-file-fixer**: Ensure files end with newline
- **check-yaml**: Validate YAML files
- **check-merge-conflict**: Prevent committing merge conflicts
- **no-commit-to-branch**: Block direct commits to main/master

Hooks run in <10 seconds for typical commits and prevent CI failures.

## Architecture

### Modular Structure (Refactored)

The codebase has been refactored into focused modules under `src/`:

- **src/metrics.py**: EvaluationMetrics dataclass and constants
- **src/strategy.py**: Strategy class, generators (rule-based & LLM), crossover operators
- **src/crucible.py**: FactorizationCrucible evaluation environment
- **src/evolution.py**: EvolutionaryEngine orchestrating the evolutionary process
- **src/comparison.py**: ComparisonEngine, BaselineStrategyGenerator, comparison logic
- **main.py**: CLI entry point with argument parsing
- **prototype.py**: Backward compatibility shim (re-exports all components)

### Core Components

1. **FactorizationCrucible**: The evaluation environment
   - Simulates GNFS sieving step
   - Evaluates strategies by counting "smooth" candidates found in fixed time
   - Uses target number N and search space around sqrt(N)

2. **EvolutionaryEngine**: Orchestrates the evolutionary process
   - Manages population of strategies across generations
   - Selection: Top 20% become parents (elite selection)
   - Reproduction (configurable rates via CLI):
     - Crossover (default 30%): Combine two elite parents via `crossover_strategies()`
     - Mutation (default 50%): Mutate single elite parent via LLM or rule-based
     - Random newcomers (default 20%): Fresh strategies for diversity
   - Uses LLMStrategyGenerator if LLM mode enabled, else basic StrategyGenerator

3. **Strategy**: Dataclass representing optimization heuristics
   - `power` (2-5): Polynomial degree for candidate generation (x^power - N)
   - `modulus_filters`: List of (modulus, residues) for fast rejection
   - `smoothness_bound`: Maximum prime factor to check (from SMALL_PRIMES)
   - `min_small_prime_hits`: Required count of small prime factors

4. **Crossover Operators** (NEW - genetic recombination):
   - `crossover_strategies(parent1, parent2)`: Uniform crossover combining two parents
     - Discrete params (power, smoothness_bound, min_hits): Random choice from either parent
     - Modulus filters: Blended via `blend_modulus_filters()`
     - Result auto-normalized via Strategy.__post_init__
   - `blend_modulus_filters(filters1, filters2)`: Intelligent filter merging
     - Merges filters with same modulus (union of residues)
     - Keeps unique filters from each parent
     - Prioritizes smaller moduli (better filtering efficiency)
     - Limits to max 4 filters
   - Fallback logic: If only one elite exists, falls back to mutation

5. **EvaluationMetrics**: Detailed metrics tracking (NEW in PR #8)
   - `candidate_count`: Total smooth candidates found
   - `smoothness_scores`: Quality metrics (lower = smoother)
   - `timing_breakdown`: Time spent in each evaluation phase
   - `rejection_stats`: Counts of rejection reasons
   - `example_candidates`: Sample smooth numbers found

6. **BaselineStrategyGenerator** (NEW - PR #14 Multi-Strategy Evaluation):
   - Generates three classical GNFS-inspired baseline strategies
   - **Conservative**: `power=2, strict filters (3), high min_hits=4` - Most selective
   - **Balanced**: `power=3, moderate filters (2), min_hits=2` - Middle ground
   - **Aggressive**: `power=4, minimal filters (1), min_hits=1` - Most permissive
   - Deterministic generation (no randomness)
   - All strategies pass validation and normalization

7. **ComparisonEngine** (NEW - PR #14):
   - Runs multiple independent evolutionary runs against baselines
   - Each run gets unique seed: `base_seed + run_index`
   - Tracks best fitness per generation (not final civilizations)
   - Integrates ConvergenceDetector for early stopping
   - Performs statistical analysis via StatisticalAnalyzer
   - Returns ComparisonRun objects with:
     - `evolved_fitness`: List[float] - Best fitness per generation
     - `baseline_fitness`: Dict[str, float] - Baseline performance
     - `generations_to_convergence`: Optional[int]
     - `final_best_strategy`: Strategy
     - `random_seed`: Optional[int]

8. **ComparisonRun**: Dataclass for single comparison run results
   - Stores complete fitness history and baseline performance
   - Includes convergence detection results
   - Serializable to JSON for analysis

### Statistical Analysis (src/statistics.py)

See README "Multi-Strategy Evaluation System" for user-facing guide. Technical details:

**StatisticalAnalyzer**: Welch's t-test, Cohen's d, 95% CI using Welch-Satterthwaite df → Returns ComparisonResult
**ConvergenceDetector**: Relative variance (variance/mean²), window=5, threshold=0.05 → Returns convergence bool/index
**ComparisonResult**: Dataclass with `interpret()` method for human-readable output

### Meta-Learning System (src/meta_learning.py, src/adaptive_engine.py)

Automatically adapts operator selection rates based on performance via UCB1 algorithm.

**Data Structures**:
- **OperatorMetadata**: Tracks `operator`, `parent_ids`, `parent_fitness`, `generation` for each civilization
- **OperatorStatistics**: Accumulates `total_offspring`, `elite_offspring`, `success_rate` per operator
- **AdaptiveRates**: Snapshot of `crossover_rate`, `mutation_rate`, `random_rate`, `operator_stats` at generation N

**MetaLearningEngine**:
- **Methods**: `update_statistics()`, `finalize_generation()`, `calculate_adaptive_rates()`, `get_operator_history()`
- **UCB1 Algorithm**: `score = success_rate + sqrt(2 * ln(total_trials) / operator_trials)`
  - Balances exploitation (success_rate) with exploration (sqrt term for less-tried operators)
  - Converts scores to rates via softmax normalization
  - Enforces bounds [0.1, 0.7], ensures sum=1.0
- **Config**: `adaptation_window=5`, `min_rate=0.1`, `max_rate=0.7`

**Integration** (EvolutionaryEngine):
1. Init: Create MetaLearningEngine if enabled
2. After elite selection: Update statistics for all civilizations
3. Before reproduction (if gen ≥ window): Calculate adaptive rates via UCB1, apply to engine
4. Offspring creation: Attach OperatorMetadata
5. Export: Include `operator_history` in metrics JSON

**Data Flow**: Evaluate → Select elites → Update stats → Finalize → (if gen≥window: adapt rates via UCB1) → Create offspring with metadata

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

## Configuration Management System

Centralized `Config` dataclass (`src/config.py`) eliminates magic numbers via single source of truth for all tunable parameters.

**Configuration Priority**: Default values → Environment variables (`.env`) → CLI arguments (highest)

**Flow**: `.env` → `load_config()` → `Config(defaults)` → CLI overrides → `__post_init__()` validates → Pass to components

### Config Parameters (23 total, 5 categories)

1. **LLM** (7): `api_key`, `max_llm_calls`, `llm_enabled`, `temperature_base`, `temperature_max`, `max_tokens`, `temperature_scaling_generations`
2. **Evolution** (4): `elite_selection_rate` (0.2), `crossover_rate` (0.3), `mutation_rate` (0.5), `evaluation_duration` (0.1)
3. **Meta-Learning** (5): `meta_learning_min_rate` (0.1), `meta_learning_max_rate` (0.7), `adaptation_window` (5), `fallback_inf_rate` (0.8), `fallback_finite_rate` (0.2)
4. **Strategy Bounds** (5): `power_min` (2), `power_max` (5), `max_filters` (4), `min_hits_min` (1), `min_hits_max` (6)
5. **Mutation Probabilities** (5): `mutation_prob_power` (0.3), `mutation_prob_filter` (0.3), `mutation_prob_modulus` (0.5), `mutation_prob_residue` (0.5), `mutation_prob_add_filter` (0.15)

### Validation (4 methods in `__post_init__()`)

1. **Evolution**: Elite rate in (0,1], crossover+mutation ≤ 1.0, duration > 0
2. **Meta-Learning**: 0 ≤ min_rate ≤ max_rate ≤ 1, 3*min_rate ≤ 1.0+ε, 3*max_rate ≥ 1.0-ε, window ≥ 1
3. **Strategy Bounds**: 2 ≤ power_min ≤ power_max ≤ 5, max_filters ≥ 1, 1 ≤ min_hits_min ≤ min_hits_max
4. **Mutation Probs**: All in [0,1]

**Epsilon**: 0.01 (1%) tolerance for floating-point comparisons (e.g., 3*0.33)

### Component Integration

All components (`EvolutionaryEngine`, `StrategyGenerator`, `MetaLearningEngine`) accept optional `config` parameter. Default `Config()` created if not provided (backward compatibility).

**CLI Integration**: 15 new arguments in `main.py` override config fields. After overrides, `config.__post_init__()` re-validates.

### Serialization

**Export**: `config.to_dict(include_sensitive=False)` excludes API keys
**Import**: `Config.from_dict(dict, api_key="...")` reconstructs from dict
**Metrics**: `export_metrics()` includes full config for reproducibility

### Design Principles

1. **Single Source of Truth**: Each parameter defined once (DRY)
2. **Fail-Fast Validation**: Invalid configs rejected at construction
3. **Security by Default**: Sensitive data excluded from export
4. **Backward Compatible**: Optional parameter with sensible defaults

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

### Test Coverage Summary

**Total Tests**: 339 comprehensive tests ensuring code quality and reliability

**Test Organization**:
1. **CLI End-to-End Tests** (27 tests) - User workflow validation via subprocess
2. **Meta-Learning Edge Cases** (25 tests) - UCB1, rate normalization, statistics
3. **Timing & Performance** (13 tests) - Overhead, extreme durations, consistency
4. **Comparison Integration** (11 tests) - Baseline strategies, RNG isolation
5. **Statistical Analysis** (24 tests) - t-tests, effect sizes, CI, convergence
6. **Core Functionality** (239 tests) - Config, strategies, evolution, crossover, etc.

### CLI End-to-End Tests (tests/test_cli.py)

**Purpose**: Validate complete user workflows via subprocess calls to main.py

**27 tests across 8 categories**:
1. **Help & Version** - Verify --help documentation
2. **Basic Workflows** - Rule-based mode, custom numbers, quiet mode
3. **JSON Export** - File creation, structure validation, security (API key exclusion)
4. **Argument Validation** - Invalid inputs, config validation, error messages
5. **Reproducibility** - --seed produces identical results
6. **Meta-Learning** - CLI flags enable features correctly
7. **Comparison Mode** - Baseline comparison workflows
8. **LLM Mode** - API key validation (conditional test if GEMINI_API_KEY set)

### Meta-Learning Edge Cases (tests/test_meta_learning_edge_cases.py)

**Purpose**: Test UCB1 algorithm, rate normalization, and statistics tracking edge cases

**25 tests across 3 categories**:
1. **TestUCB1EdgeCases** (10 tests): untried operators, zero trials, identical rates, exploration bonus
2. **TestRateNormalizationEdgeCases** (5 tests): epsilon boundaries, bounds enforcement, floating point errors
3. **TestStatisticsEdgeCases** (10 tests): negative/inf/nan fitness, large offspring counts, history alignment

**Key Findings**:
- Shallow copy pattern in get_current_statistics (acceptable for test usage)
- UCB1 exploration bonus behavior with mixed tried/untried operators
- Rate normalization with min/max bounds enforcement

### Timing & Performance Tests (tests/test_timing_accuracy.py)

**Purpose**: Validate timing measurement accuracy and performance characteristics

**13 tests across 4 categories**:
1. **TestTimingOverhead** (4 tests): overhead measurement < 50%, timing sum validation
2. **TestExtremeDurations** (3 tests): 10ms, zero, negative duration handling
3. **TestTimingConsistency** (4 tests): complexity affects timing, filter proportions
4. **TestPerformanceBenchmarks** (2 tests): baseline performance, linear scaling

**Key Findings**:
- Timing breakdown captures 60-80% of actual duration (loop overhead not captured)
- This is acceptable for performance monitoring purposes
- All timing values non-negative and sum to reasonable values

### Comparison Integration Tests (tests/test_comparison_integration.py)

**Purpose**: Validate comparison mode, baseline strategies, and convergence detection

**11 tests across 4 categories**:
1. **TestBaselineConsistency** (4 tests): deterministic evaluation, valid strategies, idempotency
2. **TestBestStrategyValidation** (2 tests): parameter validation, valid across all runs
3. **TestRNGIsolation** (3 tests): seed reproducibility, RNG independence
4. **TestConvergenceDetection** (2 tests): integration flow, window boundary

**Key Findings**:
- Timing-based evaluation means fitness not 100% deterministic
- Same seed may produce different final strategies (expected behavior)
- Tests verify no crashes and valid output across scenarios

### Statistical Edge Cases (tests/test_statistics_edge_cases.py)

**Purpose**: Test statistical analysis functions with edge cases

**24 tests across 4 categories**:
1. **TestStatisticalAnalyzer** (8 tests): identical/negative/zero distributions, variance extremes
2. **TestConvergenceDetector** (10 tests): basic/no convergence, mean near zero, insufficient history
3. **TestConfidenceIntervals** (3 tests): significance detection, variance effects
4. **TestEffectSize** (3 tests): small/large effects, zero variance handling

**Key Findings**:
- Identical distributions produce NaN p-value (expected scipy behavior)
- Small sample sizes (n=3) can produce misleading effect sizes
- Warnings about precision loss expected for identical data

**Key Patterns**:
```python
# Use sys.executable for cross-platform compatibility
import sys
cmd = [sys.executable, "main.py"] + list(args)

# Validate JSON structure after export
with open(export_path) as f:
    data = json.load(f)
assert "target_number" in data
assert "metrics_history" in data
assert "api_key" not in data["config"]  # Security check

# Test error handling
result = subprocess.run(cmd, capture_output=True, check=False)
assert result.returncode == 1
assert "ERROR" in (result.stdout + result.stderr)
```

**Local vs CI Environment Differences**:
- **Pattern**: Tests may behave differently locally vs CI due to `.env` file presence
- **Example**: `test_llm_mode_without_api_key` removes `GEMINI_API_KEY` from its test environment, but the application's startup logic reloads it from a local `.env` file if one exists
- **Solution**: This is expected behavior - tests should pass in CI (where `.env` doesn't exist) which validates the actual error handling
- **Why**: Local development legitimately uses `.env` for configuration; CI validates the error path when no configuration exists

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
2. **Install pre-commit hooks**: `make install-hooks` (one-time setup)
3. **Write tests first** for new features (TDD)
4. **Run local CI checks**: `make ci-fast` before committing
5. **Commit at logical milestones** with conventional commit format
   - Pre-commit hooks run automatically (lint, format, fast tests)
   - Hooks block commits that would fail CI
   - Fix any issues and retry commit
6. **Push to GitHub** and create PR (never push to main)
7. **Verify CI passes** before requesting review

**Pre-commit Hook Workflow**:
- Hooks run automatically on `git commit`
- If hooks fail, commit is blocked with error message
- Fix the issues (use `make format`, `make lint`, or fix tests)
- Stage fixes with `git add`
- Retry commit (hooks will re-run)
- Emergency bypass: `git commit --no-verify` (use sparingly)

## Research Validation Workflow

### Overview
Systematic hypothesis verification following pre-registered methodology to validate LLM-guided evolution vs rule-based approaches.

### Documentation Structure

**Pre-Experiment Documentation** (Phase 2):
- `research_methodology.md`: Formal hypotheses, experimental design, power analysis, pre-registered analysis plan
- `theoretical_foundation.md`: GNFS background, evolutionary algorithm theory, LLM rationale, parameter interactions
- `results_template.md`: Template for reporting results with standardized tables, figures, interpretation guidelines

**Post-Experiment Documentation** (Phase 4):
- `results_summary.md`: Executive summary with key findings, conclusions, practical recommendations
- `figures/`: Publication-quality visualizations (6 core figures: trajectories, comparisons, effect sizes, convergence, learning curves, cost-benefit)
- Filled `results_template.md`: Complete statistical analysis with all [FILL] placeholders replaced

### Experimental Phases

**Phase 1: Quick Validation** (Days 1-3)
- **Baseline validation**: 10 runs × 20 gen × 20 pop, rule-based only ($0 cost)
- **LLM proof-of-concept**: 5 runs × 15 gen × 15 pop (~$0.10-0.20)
- **Decision point**: Proceed if p<0.2 OR d>0.3 (positive signal)

**Phase 2: Documentation Sprint** (Days 4-10, parallel with Phase 1)
- Pre-register hypotheses and analysis plan BEFORE Phase 3
- Document theoretical foundation and expected mechanisms
- Prepare results template for systematic reporting

**Phase 3: Full Validation** (Days 11-21)
- **Rule-based comprehensive**: 30 runs × 30 gen × 30 pop × 1.0s ($0)
- **LLM strategic sampling**: 15 runs × 30 gen × 30 pop × 1.0s (~$1.50-2.00)
- **Meta-learning test**: 10 runs × 30 gen × 30 pop × 1.0s (~$1.00)
- **Total budget**: ~$2.50-3.50 (well under $20 limit)

**Phase 4: Analysis & Reporting** (Days 22-28)
- Aggregate results, statistical tests (Welch's t-test, Cohen's d, 95% CI)
- Generate all 6 publication-quality figures
- Fill results template, write executive summary
- Document limitations and future directions

### Statistical Analysis Workflow

**Primary Hypothesis Testing**:
```bash
# Using StatisticalAnalyzer from src/statistics.py
result = StatisticalAnalyzer().compare_fitness_distributions(
    evolved_scores=[final_fitness for run in llm_runs],
    baseline_scores=[final_fitness for run in rulebased_runs]
)
# Returns: p_value, effect_size (Cohen's d), confidence_interval, significance
```

**Decision Rules**:
- Reject H₀ if: p < 0.05 AND d ≥ 0.5 (medium effect)
- Report both statistical (p-value) and practical (effect size) significance
- Use Bonferroni correction for multiple baseline comparisons (α/3 = 0.0167)

**Convergence Analysis**:
```bash
# Using ConvergenceDetector from src/statistics.py
detector = ConvergenceDetector(window_size=5, threshold=0.05)
gen_converged = detector.generations_to_convergence(fitness_history)
```

### Reproducibility Requirements

**Seed Management**:
- Phase 1 Rule-based: seeds 42-51 (10 runs)
- Phase 1 LLM: seeds 100-104 (5 runs)
- Phase 3 Rule-based: seeds 1000-1029 (30 runs)
- Phase 3 LLM: seeds 2000-2014 (15 runs)
- Phase 3 Meta: seeds 3000-3009 (10 runs)

**Environment Specification**:
- Python 3.9+ (CI tests on 3.9, 3.10, 3.11)
- Key dependencies: google-genai>=0.2.0, scipy>=1.9.0, pydantic>=2.0.0
- Hardware: Apple Silicon M-series, 16GB+ RAM recommended

**Data Archival**:
- All results stored in `results/` directory (git-ignored)
- Naming convention: `{mode}_run_{seed}.json`
- Includes full config, metrics_history, operator_history for reproducibility

### Success Criteria

**Minimum Success** (validates framework):
- Rule-based evolution beats ≥2/3 baselines (p<0.05)
- Results reproducible (same seed → same outcome)
- Documentation complete and clear

**Target Success** (validates LLM hypothesis):
- LLM beats rule-based with d≥0.5 (medium effect)
- p<0.05 statistical significance
- Results hold across multiple baselines

**Exceptional Success** (publication-ready):
- LLM beats rule-based with d≥0.8 (large effect)
- p<0.01 strong significance
- Meta-learning shows additional benefit
- Novel strategy patterns discovered

### Common Commands

**Run experiments**:
```bash
# Phase 1: Baseline validation
python main.py --compare-baseline --num-comparison-runs 10 \
  --generations 20 --population 20 --duration 0.5 --seed 42 \
  --export-comparison results/baseline_validation.json

# Phase 1: LLM pilot
python main.py --llm --compare-baseline --num-comparison-runs 5 \
  --generations 15 --population 15 --duration 0.5 --seed 100 \
  --export-comparison results/llm_pilot.json

# Phase 3: Rule-based comprehensive (loop over seeds 1000-1029)
for seed in {1000..1029}; do
  python main.py --compare-baseline --num-comparison-runs 1 \
    --generations 30 --population 30 --duration 1.0 --seed $seed \
    --export-comparison results/rulebased_run_${seed}.json
done

# Phase 3: LLM strategic sampling (loop over seeds 2000-2014)
for seed in {2000..2014}; do
  python main.py --llm --compare-baseline --num-comparison-runs 1 \
    --generations 30 --population 30 --duration 1.0 --seed $seed \
    --export-comparison results/llm_run_${seed}.json
done
```

**Analyze results**:
```bash
# Create aggregation script
python scripts/aggregate_results.py

# Open Jupyter notebooks for visualization
jupyter notebook analysis/visualize_comparison.ipynb
```

### Critical Validation Checks

Before declaring experiments complete:
- [ ] All planned runs completed (check N_valid = N_planned)
- [ ] No systematic failures (failed runs <10%)
- [ ] Normality assumptions checked (Shapiro-Wilk test)
- [ ] Outliers identified and policy documented
- [ ] All statistical tests run with correct parameters
- [ ] All figures generated and interpretable
- [ ] Results template completely filled (no [FILL] remaining)
- [ ] Conclusions directly answer pre-registered hypotheses

## Critical Learnings

### Temperature Calculation & LLM Patterns (PR #2)
- **Temperature Inversion**: Original increased temp (0.8→1.2) over generations instead of decreasing, inverting exploration→exploitation tradeoff
- **Fix**: `temp = temp_max - (temp_max - temp_base) * progress` (starts high, decreases to base)
- **Finally Block for Counters**: Use `finally` to increment API call counters, prevents resource leaks on failures
- **Early Validation**: Validate config (API keys) at load time with clear error messages (fail-fast)

### Genetic Algorithms & Randomness (PR #10, #12)
- **Parent Selection**: Use `random.sample(elites, 2)` not `random.choice()` twice - prevents selecting same parent (asexual reproduction), critical for diversity
- **Filter Merging**: Use `sorted(set(list1) | set(list2))` not `sorted(set(list1 + list2))` - cleaner set operations
- **RNG Seeding in Tests**: Let components handle seeding (`Engine(seed=42)` only), don't seed externally - tests verify component actually works
- **Single Seeding Location**: Seed only in `__init__` where randomness used, not in `main()` - second call overwrites first

### Critical Bug Fixes (PR #14)
- **Returning Unevaluated Strategy**: `final_best_strategy` selected AFTER `run_evolutionary_cycle()` replaced civilizations with next gen (fitness=0)
  - Fix: Return `(best_fitness, best_strategy)` tuple BEFORE mutation, updated 8 call sites
  - Changed signature: `def run_evolutionary_cycle() -> float` → `-> tuple[float, Strategy]`
- **ZeroDivisionError**: `improvement_pct = ((evolved / baseline) - 1) * 100` when baseline=0
  - Fix: `... if baseline > 0 else float("inf")`, added regression test

### Testing Patterns (PR #18, #25)
- **Warning Testing**: Mock to force rare conditions → `pytest.warns()` → verify message, use `stacklevel=2` in `warnings.warn()`
- **Bounds Validation**: Test validators with `pytest.raises(ValueError, match="regex")` - validation exists but needs tests
- **Mutation Prevention**: Return copies from getters (`self.stats.copy()`) - prevents external mutation of internal state
- **Test Isolation**: Use `autouse` pytest fixture to reset global state (loggers, RNG) before/after each test - prevents pollution

### Modular Refactoring & Architecture (PR #21)
- **Zero Breaking Changes**: Bottom-up extraction (foundation → dependent → high-level) + re-export shim (`from src.X import Y` in old file)
  - Split prototype.py (1449 lines) → 6 modules, all 164 tests passing
- **Conditional Imports**: Import heavy deps only when needed - `if args.feature: from heavy_module import Class`
  - scipy only loaded for comparison mode, basic mode runs without it
- **CLI Validation**: Check core params after parsing - `if args.param < 1: sys.exit("❌ ERROR: param must be >= 1")`
- **Semantic Consistency**: Edge cases behave identically across code paths - `candidate==0` returns False in both simple and detailed evaluation

### Configuration Management (PR #23)
- **Config Propagation**: Pass same config to all components in tests - extract to variable first, prevents mismatches
- **Immutable Pattern**: `Config.from_args_and_env()` factory method constructs once with validation - no post-hoc mutation
- **ClassVar Serialization (Python 3.9-3.10)**: Explicitly `pop("EPSILON")` from `to_dict()` - `asdict()` incorrectly includes ClassVars

### Test Coverage Expansion Strategy (PR #28, #30)
- **Systematic Edge Case Testing**: Create dedicated test files per concern (meta-learning, timing, comparison, statistics)
- **Shared Fixtures Pattern**: Use conftest.py for reusable fixtures (test_crucible, test_number) - DRY principle
- **Test Organization**: Group related tests into classes with descriptive names
  - Example: `TestTimingOverhead`, `TestExtremeDurations`, `TestTimingConsistency`
- **Edge Case Categories**: Boundary conditions, extreme values, error cases, performance validation
- **Test Count Evolution**: 164 tests (PR #27) → 339 tests (PR #30) - 175 new edge case tests
- **Production Bug Discovery**: Tests found smoothness_bound normalization bug - not always from SMALL_PRIMES list

### Pre-commit Hooks & Local CI (PR #33)
- **Flaky Test Exclusion**: Exclude environment-dependent tests from hooks with `--deselect`
  - Environment state tests (API key presence/absence)
  - Timing-dependent tests (performance benchmarks with variance)
  - Integration tests (marked, run in full CI only)
- **Hook Performance**: `always_run: false` + `-m "not integration"` keeps hooks <10s
- **AI Reviewer Verification**: Gemini claimed all package versions invalid - ALL verified correct via GitHub API/PyPI
  - Pattern: Always verify factual claims (versions, dates, paths) before accepting
  - Correctness > Compliance - reject false positives even from authoritative sources
- **Transitive Dependencies**: pyyaml available via jupyter dependency chain, no explicit requirement needed
- **Makefile-based CI**: Convenient targets (`make ci`, `make ci-fast`) reduce friction for local validation

### Prometheus Phase 1 Integration & Benchmarking (PR #39)
- **Resource Initialization Pattern**: Always check state before initializing stateful resources
  - **Critical Bug**: `tracemalloc.start()` crashes with RuntimeError if called when already tracing
  - **Fix**: `if not tracemalloc.is_tracing(): tracemalloc.start()`
  - **Applies to**: File handles, database connections, any non-idempotent initialization
  - **Pattern**: Check state → initialize → track lifecycle
- **Test Reproducibility Documentation**: Document what tests verify vs what users assume
  - Don't claim "consistent results" when tests only verify "structural validity"
  - Timing-based evaluation means fitness varies - test structure (strategy params, message counts)
  - Pattern: Be explicit about documented variance in test docstrings
- **Zero Fitness Edge Cases**: Accept legitimate edge cases in timing-based evaluation
  - search_only/eval_only modes may produce zero fitness with unlucky random strategies
  - Solution: `assert fitness >= 0` not `fitness > 0`
  - Applies whenever: random initialization + short evaluation duration + strict criteria
- **Magic Number Documentation**: All magic numbers → named constants with explanatory comments
  - Example: Seed offsets (0, 1000, 2000, 3000) → `MODE_SEED_OFFSETS` dict with RNG independence comment
  - Pattern: Extract constant, add comment explaining why these specific values
- **Review Iteration Efficiency**: Group fixes by priority, test locally, push once per iteration
  - PR #39: 3 iterations (initial fixes → CI fix → tracemalloc fix) = efficient resolution
  - Pattern: Critical → medium → low priority; run full test suite locally before push
