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

**Comparison mode** (statistical analysis against baselines):
```bash
# Rule-based comparison with 5 independent runs
python prototype.py --compare-baseline --num-comparison-runs 5 --generations 10 --population 10 --seed 42

# LLM-guided comparison with JSON export
python prototype.py --llm --compare-baseline --num-comparison-runs 3 --generations 5 \
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

**StatisticalAnalyzer**:
- **Welch's t-test**: Compares fitness distributions without assuming equal variances
- **Cohen's d effect size**: Quantifies practical significance
  - d < 0.2: Negligible
  - 0.2 ≤ d < 0.5: Small
  - 0.5 ≤ d < 0.8: Medium
  - d ≥ 0.8: Large
- **95% Confidence Intervals**: Uses Welch-Satterthwaite degrees of freedom
- Returns ComparisonResult with all metrics

**ConvergenceDetector**:
- Detects fitness plateaus using rolling window variance
- Algorithm: Relative variance = variance / mean² (coefficient of variation squared)
- Configurable window size (default: 5 generations) and threshold (default: 0.05)
- `has_converged(fitness_history)`: Returns bool
- `generations_to_convergence(fitness_history)`: Returns generation index or None
- Handles edge cases: near-zero mean, insufficient data

**ComparisonResult**: Dataclass with interpretation
- Contains all statistical metrics (means, p-value, effect size, CI, significance)
- `interpret()`: Human-readable string explaining results
- Automatically categorizes effect sizes and formats output

### Meta-Learning System (src/meta_learning.py, src/adaptive_engine.py)

**Purpose**: Automatically adapts operator selection rates based on performance, eliminating manual hyperparameter tuning.

**OperatorMetadata** (Dataclass):
- Tracks how each strategy was created
- Fields: `operator` (crossover/mutation/random), `parent_ids`, `parent_fitness`, `generation`
- Attached to each civilization in `civilizations` dict
- Used to calculate fitness improvement: `fitness - parent_fitness`

**OperatorStatistics** (Dataclass):
- Accumulated performance metrics per operator
- Fields: `total_offspring`, `elite_offspring`, `total_fitness_improvement`, `avg_fitness_improvement`, `success_rate`
- Updated after each generation's elite selection
- Success rate = `elite_offspring / total_offspring`

**AdaptiveRates** (Dataclass):
- Snapshot of rates and statistics at a given generation
- Fields: `crossover_rate`, `mutation_rate`, `random_rate`, `generation`, `operator_stats`
- Returned by `calculate_adaptive_rates()` method

**MetaLearningEngine** (Class):
- Manages operator performance tracking and rate adaptation
- **Key Methods**:
  - `update_statistics(operator, fitness_improvement, became_elite)`: Records offspring performance
  - `finalize_generation()`: Saves current stats to history, prepares for next generation
  - `calculate_adaptive_rates(current_rates)`: Computes new rates using UCB1 algorithm
  - `get_operator_history()`: Returns historical statistics for all generations
- **UCB1 Algorithm** (Upper Confidence Bound):
  - Formula: `score = success_rate + sqrt(2 * ln(total_trials) / operator_trials)`
  - Balances exploitation (high success rate) with exploration (sqrt term for less-tried operators)
  - Converts scores to rates via softmax-like normalization
  - Enforces rate bounds: [min_rate=0.1, max_rate=0.7]
  - Ensures sum to 1.0 via renormalization
- **Configuration**: `adaptation_window` (default: 5), `min_rate`, `max_rate`

**Integration with EvolutionaryEngine**:
1. **Initialization** (line ~650): Creates MetaLearningEngine if `enable_meta_learning=True`
2. **Population Init** (line ~670): Adds OperatorMetadata to each civilization
3. **After Elite Selection** (line ~757): Updates statistics for all civilizations
4. **Before Reproduction** (line ~780): Calculates and applies adaptive rates
5. **Offspring Creation** (line ~870): Attaches OperatorMetadata to new civilizations
6. **Export** (line ~907): Includes `operator_history` in metrics JSON

**Data Flow**:
```
Generation N:
1. Evaluate all civilizations → assign fitness
2. Select elites (top 20%)
3. Update operator statistics (which operators created elites?)
4. Finalize generation (save to history)
5. IF generation >= adaptation_window:
     Calculate adaptive rates using UCB1
     Update self.crossover_rate, self.mutation_rate, self.random_rate
6. Create next generation with current rates
7. Attach operator metadata to offspring
```

**Example Adaptation**:
```
Generation 0-4: Fixed rates (0.3/0.5/0.2)
Generation 5:
  - Crossover: 67% elite rate (4/6 offspring) → high UCB score
  - Mutation: 33% elite rate (2/6 offspring) → medium UCB score
  - Random: 0% elite rate (0/2 offspring) → low UCB score
  - New rates: 0.52 crossover, 0.28 mutation, 0.20 random
```

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

## Critical Learnings from PR #10

1. **Parent Selection in Genetic Algorithms**: Use `random.sample(population, k)` not multiple `random.choice()` calls
   - Issue: Calling `random.choice(elites)` twice can select the same parent (asexual reproduction)
   - Solution: `parent1, parent2 = random.sample(elites, 2)` ensures distinct parents
   - Impact: Critical for genetic diversity, especially with small populations

2. **Filter Merging Optimization**: Use set union operators instead of list concatenation + deduplication
   - Before: `sorted(set(list1 + list2))`
   - After: `sorted(set(list1) | set(list2))`
   - Impact: More readable and slightly more efficient

## Critical Learnings from PR #12

1. **RNG Seeding in Tests**: Let components handle their own seeding
   - Issue: External `random.seed()` in tests masks whether component's seeding works
   - Wrong: `random.seed(42); engine = Engine(seed=42)` - test passes even if engine doesn't seed
   - Correct: `engine = Engine(seed=42)` - test verifies engine actually applies seed
   - Impact: Tests verify actual user experience, catch seeding bugs in components

2. **Duplicate RNG Seeding Anti-Pattern**: Seed only in one place
   - Issue: Seeding in main() and __init__() - second call overwrites first
   - Wrong: `random.seed(args.seed)` in main(), then `random.seed(seed)` in __init__
   - Correct: Seed only in __init__ where component uses randomness
   - Impact: Eliminates confusion, ensures seeding happens correctly for all usage patterns

## Critical Learnings from PR #14

1. **Systematic Multi-Issue PR Review Handling**: Fix all issues in priority order
   - Received 7 code review issues from gemini-code-assist and chatgpt-codex-connector
   - Prioritized: 2 HIGH (ZeroDivisionError fixes), 1 CRITICAL (final_best_strategy bug), 4 MEDIUM (code quality)
   - Fixed all in single commit with comprehensive testing
   - Pattern: Discover → Prioritize → Fix by priority → Extract duplicates → Verify
   - Success: 126 tests passing, all CI green

2. **Critical Bug: Returning Unevaluated Strategy**: Capture state BEFORE mutation
   - Issue: `final_best_strategy` selected from `engine.civilizations` AFTER `run_evolutionary_cycle()` completed
   - Root cause: `run_evolutionary_cycle()` replaces civilizations with next generation (fitness=0, unevaluated)
   - Wrong: `max(engine.civilizations.items(), key=lambda x: x[1]["fitness"])` after cycle → random unevaluated strategy
   - Correct: Return `(best_fitness, best_strategy)` tuple BEFORE replacing civilizations
   - Impact: Fixed major bug where exported ComparisonRun had wrong strategy; updated 8 call sites

3. **Edge Case Handling: ZeroDivisionError in Statistical Analysis**
   - Issue 1: `improvement_pct = ((evolved_mean / baseline_mean) - 1) * 100` when baseline_mean=0 (conservative finds no candidates)
   - Fix: `improvement_pct = ... if baseline_mean > 0 else float("inf")`
   - Issue 2: Welch-Satterthwaite df calculation when both variances zero: `df = num / den` when den=0
   - Fix: `df = num / den if den > 0 else float(n1 + n2 - 2)`
   - Test: Added `test_comparison_result_interpret_zero_baseline()` to prevent regression

4. **DRY Principle Application**: Extract duplicate code immediately when spotted
   - Duplicate 1: LLM cost summary (6 lines × 2 occurrences) → `print_llm_summary(llm_provider)` helper
   - Duplicate 2: Convergence logic (20+ lines in 2 methods) → `_is_window_converged(window)` private method
   - Impact: Reduces maintenance burden, single source of truth
   - Timing: Fix during PR review, not "later"

5. **Function Signature Evolution**: Update ALL call sites when changing return type
   - Changed: `def run_evolutionary_cycle() -> float` → `-> tuple[float, Strategy]`
   - Required: Update 8 call sites (main() + 4 test files)
   - Method: `grep -r "run_evolutionary_cycle()"` to find all occurrences
   - Pattern: `best_fitness, best_strategy = engine.run_evolutionary_cycle()`
   - Why: Better to break at compile time than silently use wrong value
## Critical Learnings from PR #16

1. **Jupyter Notebook Edge Case Robustness**: Always defend against unexpected data conditions
   - Use `.get()` with defaults for JSON keys: `runs = data.get("runs", [])`
   - Check empty collections before operations: `if not all_evolved: ... else: max_gens = max(..., default=0)`
   - Handle None before formatting: `if conv_stats['mean'] is not None: ... else: "No runs converged"`
   - Conditional expressions for division by zero: `improvement_pct = ... if baseline_mean > 0 else float("inf")`
   - Impact: Notebooks handle edge cases gracefully (0% convergence, baseline=0, empty runs)
   - Why: Experimental data often has unexpected conditions; robust handling enables exploration

2. **Systematic PR Review Response**: Address all feedback comprehensively in priority order
   - PR #16: Fixed 5 robustness issues from gemini-code-assist and chatgpt-codex-connector
   - Issues: KeyError, ValueError (empty runs), IndexError (runs[0]), ValueError (ci_upper), TypeError (None formatting)
   - Pattern: Read ALL review comments → Prioritize by severity → Fix systematically → Test edge cases → Commit
   - Result: All edge cases tested and passing, documentation complete

3. **results/ Directory Documentation**: Document data directories even if gitignored
   - Created `results/README.md` explaining purpose, usage, file format
   - Added directory creation instruction to main README: `mkdir -p results`
   - Why: Users need to know what directories are for and how to create them
   - Impact: Prevents FileNotFoundError when following README examples

## Critical Learnings from PR #18

1. **GraphQL for Complete PR Feedback Coverage**: Use GraphQL single query instead of multiple REST calls
   - Created `/tmp/pr_feedback_query.gql` with atomic query fetching comments, reviews, line comments, CI annotations
   - Replaced broken REST approach (404 errors on empty collections) with 100% reliable coverage
   - Pattern: Single GraphQL query → Parse JSON → Filter by timestamp
   - Why: REST API returns 404 for empty review collections; GraphQL returns empty arrays
   - Impact: Zero missed reviewer feedback, eliminated false negatives

2. **Incremental Post-Commit Review Handling**: Always check for new feedback after pushing fixes
   - Pattern: Fix issues → Commit → Push → Wait for CI → Check for NEW feedback since commit
   - Use `/fix_pr_since_commit_graphql` to filter feedback by timestamp vs second-to-last commit
   - Why: Reviewers (especially AI) can post feedback AFTER you push fixes
   - Example: PR #18 received 6 new comments from claude-review after initial fixes pushed
   - Impact: Prevented merge with unaddressed feedback

3. **Warning Testing Pattern**: Test warning mechanisms with mock patching
   - Pattern: Create mock that forces non-convergence → Use `pytest.warns()` → Verify warning message
   - Example: `test_enforce_rate_bounds_convergence_warning` patches method to force max iterations
   - Use `stacklevel=2` in `warnings.warn()` for proper source tracking (prevents B028 lint error)
   - Why: Convergence failures are rare in normal operation but must be tested
   - Impact: Verified warning mechanism works without waiting for pathological inputs

4. **Bounds Validation Testing**: Always test initialization validators
   - Added 3 tests: `min > max`, `3*min > 1.0`, `3*max < 1.0`
   - Pattern: `pytest.raises(ValueError, match="regex")` for validation errors
   - Why: Validation exists but wasn't tested - could silently break
   - Impact: Caught feasibility bugs before production use

5. **Mutation Prevention in Getters**: Return copies from getter methods
   - Issue: `get_current_statistics()` returned direct reference, allowing external mutation
   - Fix: `return self.current_stats.copy()` prevents callers from modifying internal state
   - Why: Defensive programming - internal state should be immutable from outside
   - Impact: Eliminated subtle mutation bugs
