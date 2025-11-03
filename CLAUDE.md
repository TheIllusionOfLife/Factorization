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

## Configuration Management System (src/config.py)

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
- Use `sys.executable` for cross-platform subprocess calls
- Validate JSON structure after export (check keys, exclude sensitive data)
- Test error handling with `capture_output=True, check=False` pattern
- Local vs CI differences: Tests may behave differently due to `.env` file presence (expected)

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

## Research Validation Workflow

Systematic hypothesis verification using pre-registered methodology to validate LLM-guided vs rule-based evolution. See `docs/research_methodology.md` for complete protocols.

**Experimental Phases**: Quick validation (10-15 runs) → Documentation sprint → Full validation (30 runs rule-based, 15 LLM, budget ~$3) → Analysis & reporting

**Statistical Analysis**: Use `StatisticalAnalyzer` and `ConvergenceDetector` from `src/statistics.py` for Welch's t-test, Cohen's d, 95% CI. Decision rule: Reject H₀ if p<0.05 AND d≥0.5.

**Reproducibility**: Seed ranges by phase (Phase 1: 42-104, Phase 3: 1000-3009). Python 3.9+, key deps: google-genai≥0.2.0, scipy≥1.9.0. Results in `results/` as `{mode}_run_{seed}.json`.

**Success Criteria**: Minimum (rule-based beats ≥2/3 baselines, p<0.05) | Target (LLM beats rule-based, d≥0.5, p<0.05) | Exceptional (d≥0.8, p<0.01, novel patterns)

**Common Commands**: See `docs/research_methodology.md` for experiment run commands and analysis workflow

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
- **Resource Initialization**: Check state before init: `if not tracemalloc.is_tracing(): tracemalloc.start()` (prevents RuntimeError on re-init)
- **Test Documentation**: Document timing variance as expected behavior, not claims of exact reproducibility
- **Zero Fitness Edge Cases**: `assert fitness >= 0` not `> 0` (random init + short duration can legitimately produce zero)
- **Magic Numbers**: Extract to named constants with explanatory comments (e.g., `MODE_SEED_OFFSETS` for RNG independence)

### Prometheus Timing Variance Investigation (PR #42)
- **Timing Variance NOT a Bug**: Short evaluation (0.1s) + random init + unlucky RNG can legitimately produce 0 fitness
- **Solution**: Use longer durations (0.5-1.0s) in regression tests to test correctness, not timing sensitivity
- **Regression Tests**: Added 4 tests with varied durations to prevent false bug reports

### C1 Validation and Cross-Version Testing (PR #47)
- **Python RNG Cross-Version**: Seed 42 fails on Python 3.9 but works on 3.10+. Solution: Use seed 100 that works across all versions. Never assume seed reproducibility across Python versions in timing-based tests.
- **Strategy.copy() Config Propagation**: Use `strategy.copy()` not `copy.deepcopy()` to preserve `_config` field needed for normalization in `__post_init__()`.
- **Feedback-Guided Mutations**: Keyword parsing (slow→speed, low fitness→coverage, low smoothness→quality, good→refinement) maps feedback to 4 mutation strategies with clear trade-offs.
- **Statistical Testing**: Emergence factor = collaborative_mean/max(baselines), requires >1.1 + p<0.05 + d≥0.5. H1a result: 0.95, p=0.58, d=-0.58 → NOT supported (collaborative underperformed by 4.6%).
- **JSON Compatibility**: Graceful fallback checking `metrics_history` → `best_fitness` → `final_fitness` with warnings, supports evolving formats.
- **Test Thresholds**: Increased ratio from 100x to 500x based on empirical RNG variance in small runs, detects bugs while tolerating legitimate variance.

### C1 Results Analysis & Visualization (PR #49)
- **Publication-Quality Figures**: Create standalone visualization script with 6 core figures (fitness comparison, distributions, emergence factor, statistical tests, hypothesis summary, baseline comparison) - export both PNG (300 DPI) and SVG for publication
- **Comprehensive Results Summary**: Executive summary document (C1_RESULTS_SUMMARY.md) with 11 sections: Executive Summary, Experimental Design, Results, Interpretation, Recommendations, Limitations, Data Availability, Conclusion, References
- **Interactive Exploration Notebook**: Jupyter notebook with 11 analysis sections covering descriptive stats, distributions, normality testing, statistical tests, CI, emergence analysis, power analysis, hypothesis summary
- **Negative Result Value**: Scientifically valuable negative results require clear documentation of root causes, not just "hypothesis not supported" - explain WHY (Phase 1 MVP validated infrastructure but deferred feedback integration to Phase 2)
- **Power Analysis Reporting**: Always include retrospective power analysis in results summary - C1 achieved 45% power (underpowered for α=0.05, d=0.5), required N=28 per group for 80% power with observed effect
- **Multiple Deliverables Structure**: Separate visualization script (automated), results summary (human-readable), and exploration notebook (interactive) serve different audiences and use cases
- **Root Cause Integration**: Reference detailed root cause analysis document (`docs/prometheus_underperformance_analysis.md`) from results summary for full technical context
- **Critical Data File Tracking**: Always verify generated data files (like h1a_analysis.json) are tracked in git, not just exist locally - scripts depend on these files and will fail without them
- **Dependency Completeness for Notebooks**: Check all notebook imports and ensure dependencies are in requirements-dev.txt (e.g., statsmodels for power analysis)
- **Figure Filename Consistency**: Documentation must reference actual generated filenames - verify script output matches markdown references (e.g., figure2_distribution_analysis.png not figure2_distributions.png)
