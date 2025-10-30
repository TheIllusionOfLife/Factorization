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

### CLI End-to-End Tests (tests/test_cli.py) - NEW

**Purpose**: Validate complete user workflows via subprocess calls to main.py

**26 tests across 8 categories**:
1. **Help & Version** - Verify --help documentation
2. **Basic Workflows** - Rule-based mode, custom numbers, quiet mode
3. **JSON Export** - File creation, structure validation, security (API key exclusion)
4. **Argument Validation** - Invalid inputs, config validation, error messages
5. **Reproducibility** - --seed produces identical results
6. **Meta-Learning** - CLI flags enable features correctly
7. **Comparison Mode** - Baseline comparison workflows
8. **LLM Mode** - API key validation (conditional test if GEMINI_API_KEY set)

**Key Patterns**:
```python
# Use sys.executable for cross-platform compatibility
import sys
cmd = [sys.executable, "main.py"] + list(args)

# Validate JSON structure after export
data = json.load(open(export_path))
assert "target_number" in data
assert "metrics_history" in data
assert "api_key" not in data["config"]  # Security check

# Test error handling
result = subprocess.run(cmd, capture_output=True, check=False)
assert result.returncode == 1
assert "ERROR" in (result.stdout + result.stderr)
```

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

## Docker Security Patterns

### Security Layered Defense
**When**: Running untrusted code (LLM-generated)
**Pattern**: Combine 4 isolation layers
```python
container = client.containers.create(
    image="sandbox:latest",
    network_disabled=True,  # Layer 1: Network isolation
    read_only=True,  # Layer 2: Filesystem protection
    tmpfs={"/tmp": "size=100m,uid=1000"},  # Writable tmpfs for numpy
    mem_limit="512m", memswap_limit="512m",  # Layer 3: Resource limits
    cpu_period=100000, cpu_quota=50000, pids_limit=100,
    user="1000:1000",  # Layer 4: Non-root user
)
```

### Container Cleanup
**Pattern**: Always cleanup in finally block with force removal
```python
container = None
try:
    container = client.containers.create(...)
    container.start()
    exit_info = container.wait(timeout=timeout)
    logs = container.logs()
except Exception:
    if container:
        container.kill()
        container.wait()
    raise
finally:
    if container is not None:
        try:
            container.remove(force=True)
        except Exception:
            pass  # Ignore cleanup errors
```

### OOM Kill Detection
**Pattern**: Check `inspect["State"].get("OOMKilled", False)` after wait
**Why**: OOM kills indicate resource config issue, not code bug

### Base64 + Pickle Serialization
**When**: Passing numpy arrays to containers
**Pattern**: `pickle.dumps() → base64.encode() → embed in script → decode → pickle.loads()`
```python
# Host
grid_b64 = base64.b64encode(pickle.dumps(numpy_array)).decode("ascii")
script = f'grid_b64 = {repr(grid_b64)}\nnumpy_array = pickle.loads(base64.b64decode(grid_b64))'
# Result: print(f"RESULT:{{base64.b64encode(pickle.dumps(result)).decode()}}")
result = pickle.loads(base64.b64decode(logs.split("RESULT:")[1].strip()))
```
**Benefits**: No volume mounts, works with read-only filesystem
**Security**: Only for trusted container data

### Bandit False Positives
**Pattern**: `# nosec` with explanatory comment
```python
tmpfs={"/tmp": "size=100m,uid=1000"},  # noqa: S108  # nosec - Docker tmpfs, ephemeral
result = pickle.loads(result_bytes)  # noqa: S301  # nosec - trusted container
```
**Use for**: `/tmp` in Docker, `pickle` from trusted sources, `subprocess` with controlled input

## Evolution System Patterns

### Accurate Evaluation Metrics
**Pattern**: `new_evaluations = len(population) - config.elite_size` (elite preservation doesn't count as new evaluation)

### State vs Transition Clarity
**Pattern**: Document if number = completed state or next transition (prevents off-by-one errors)

### Consistent Operation Timing
**Pattern**: Standardize timing (e.g., snapshots AFTER evolution, not before)

### Enhanced Semantic Operators Pattern
**Concept**: High-scoring ideas trigger revolutionary variations (breakthrough mutations)
**Implementation**:
```python
# Breakthrough threshold
if fitness >= 0.8:
    mutation_type = "breakthrough"
    temperature = 0.95  # Higher creativity
    max_tokens = default_tokens * 2  # More space for creative exploration
else:
    mutation_type = "regular"
    temperature = 0.8
    max_tokens = default_tokens
```
**Cache Handling**: Store mutation type alongside content for consistency
**Validation**: Always validate cache has "content" key before using
**Batch Consistency**: Never skip ideas in batch - fallback to original content if cache invalid
**Type Safety**: Initialize variables before conditional blocks to avoid mypy errors

### Semantic Diversity Implementation
**Pattern**: Use Gemini embeddings API with caching + Jaccard fallback
```python
class GeminiDiversityCalculator:
    async def calculate_diversity(self, population):
        embeddings = await self._get_embeddings_with_cache(texts)  # :batchEmbedContents
        similarity_matrix = cosine_similarity(embeddings)
        return 1.0 - np.mean(upper_triangle)

# Always provide fallback
if diversity_method == SEMANTIC:
    primary = GeminiDiversityCalculator(llm)
    fallback = JaccardDiversityCalculator()
```

### Architecture Unification Pattern
**When**: Multiple components share similar functionality with duplicated code
**How**: Extract shared functionality to base class, use inheritance
**Example**: PR #158 - Created BatchOperationsBase to eliminate ~180 lines of duplicate batch processing code
```python
class BatchOperationsBase:
    def prepare_advocacy_input(self, candidates):
        return [{"idea": c["text"], "evaluation": c["critique"]} for c in candidates]

class AsyncCoordinator(BatchOperationsBase):
    def __init__(self):
        super().__init__()  # Initialize base class
```
**Benefits**: Single source of truth, easier testing, consistent behavior
