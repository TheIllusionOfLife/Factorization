# Codebase Review and Refactoring Plan

**Date**: 2025-11-05
**Scope**: Complete analysis of Factorization codebase
**Total LOC**: ~4,335 lines (src/ directory)
**Files Analyzed**: 15 source files across 3 modules

---

## Executive Summary

The codebase demonstrates solid architectural patterns with modular design, comprehensive testing, and good documentation. However, several critical issues require attention:

1. **Critical**: Extensive use of Japanese comments (3,055+ lines) in production code
2. **High**: Code duplication in baseline evaluation logic
3. **High**: Configuration verbosity (87-line if-statement chain)
4. **Medium**: Long functions exceeding maintainability thresholds
5. **Medium**: Embedded prompt strings reducing testability
6. **Low**: Minor type hint gaps

**Overall Assessment**: The codebase is functional and well-tested, but needs significant internationalization cleanup and refactoring for long-term maintainability.

---

## Issue Categories

### 1. CRITICAL: Language Mixing (Japanese Comments)

**Impact**: High - Reduces code accessibility, violates internationalization best practices
**Scope**: Widespread - 3,055+ lines affected across all major modules

#### Affected Files:
- `src/evolution.py` - Class docstrings and inline comments
- `src/strategy.py` - Method documentation (LLMStrategyGenerator)
- All other source files contain Japanese characters

#### Examples:

```python
# evolution.py:24
"""
文明の世代交代を司る。優れた戦略を選択し、次世代の戦略を生み出す。
"""

# evolution.py:85
"""最初の文明（戦略）群を生成する"""

# strategy.py:279-280
"""
LLM統合版の戦略生成器。
LLMによる戦略提案を試み、失敗時は従来のルールベース手法にフォールバックする。
"""
```

#### Recommendation:
**Action**: Replace ALL Japanese comments with English equivalents
**Priority**: CRITICAL - Do this FIRST before any other refactoring
**Effort**: High (3-4 hours for translation + review)
**Risk**: Low - Comments only, no logic changes

**Implementation Plan**:
1. Use automated translation tools for first pass
2. Manual review by bilingual developer
3. Verify no functional changes with full test suite
4. Update CLAUDE.md to prohibit non-English comments in style guide

---

### 2. HIGH: Code Duplication in Baseline Evaluation

**Impact**: High - Violates DRY principle, increases maintenance burden
**Scope**: `src/prometheus/experiment.py` - 3 similar baseline implementations

#### Problem:
The `run_independent_baseline()` method contains 130+ lines with three nearly identical code paths for `search_only`, `eval_only`, and `rulebased` modes.

#### Code Smell:
```python
# Lines 317-349: search_only baseline
search_agent = SearchSpecialist(...)
for _ in range(generations):
    for _ in range(population_size):
        strategy = search_agent.strategy_generator.random_strategy()
        metrics = self.crucible.evaluate_strategy_detailed(...)
        fitness = metrics.candidate_count
        if fitness > best_fitness:
            best_fitness = fitness
            best_strategy = strategy

# Lines 351-391: eval_only baseline (nearly identical)
eval_agent = EvaluationSpecialist(...)
generator = StrategyGenerator(...)
for _ in range(generations):
    for _ in range(population_size):
        strategy = generator.random_strategy()
        # ... same evaluation logic
```

#### Recommendation:
**Action**: Extract common pattern into helper method
**Priority**: HIGH
**Effort**: Medium (2-3 hours)

**Proposed Refactoring**:

```python
def _run_baseline_with_strategy_source(
    self,
    strategy_source: Callable[[], Strategy],
    generations: int,
    population_size: int,
    evaluator: Optional[EvaluationSpecialist] = None,
) -> Tuple[float, Strategy]:
    """
    Generic baseline runner that accepts any strategy generation source.

    Args:
        strategy_source: Callable that returns a Strategy
        generations: Number of generations
        population_size: Population size
        evaluator: Optional evaluator agent (if None, use crucible directly)

    Returns:
        Tuple of (best_fitness, best_strategy)
    """
    best_fitness = 0.0
    best_strategy: Optional[Strategy] = None

    for _ in range(generations):
        for _ in range(population_size):
            strategy = strategy_source()

            if evaluator:
                msg = Message(...)
                response = evaluator.process_request(msg)
                fitness = response.payload["fitness"]
            else:
                metrics = self.crucible.evaluate_strategy_detailed(
                    strategy, self.config.evaluation_duration
                )
                fitness = metrics.candidate_count

            if fitness > best_fitness:
                best_fitness = fitness
                best_strategy = strategy

    # Fallback logic
    if best_strategy is None:
        warnings.warn(...)
        best_strategy = strategy_source()

    return best_fitness, best_strategy
```

Then simplify baselines:

```python
if agent_type == "search_only":
    agent = SearchSpecialist(...)
    return self._run_baseline_with_strategy_source(
        strategy_source=lambda: agent.strategy_generator.random_strategy(),
        generations=generations,
        population_size=population_size,
    )

elif agent_type == "eval_only":
    agent = EvaluationSpecialist(...)
    gen = StrategyGenerator(...)
    return self._run_baseline_with_strategy_source(
        strategy_source=gen.random_strategy,
        generations=generations,
        population_size=population_size,
        evaluator=agent,
    )
```

**Benefits**:
- Reduces 130 lines to ~70 lines (46% reduction)
- Single source of truth for evaluation logic
- Easier to add new baseline types
- Improved testability

---

### 3. HIGH: Configuration Verbosity

**Impact**: Medium - Reduces readability, prone to copy-paste errors
**Scope**: `src/config.py:222-299` (87 lines of repetitive if-statements)

#### Problem:
The `Config.from_args_and_env()` method contains 87 lines of nearly identical if-statement patterns:

```python
if args.duration is not None:
    overrides["evaluation_duration"] = args.duration
if args.elite_rate is not None:
    overrides["elite_selection_rate"] = args.elite_rate
if args.crossover_rate is not None:
    overrides["crossover_rate"] = args.crossover_rate
# ... 15 more identical patterns
```

#### Recommendation:
**Action**: Use data-driven mapping approach
**Priority**: HIGH
**Effort**: Low (1 hour)

**Proposed Refactoring**:

```python
@classmethod
def from_args_and_env(cls, args, use_llm: bool) -> "Config":
    """Create Config from CLI args and environment variables."""
    # ... (existing setup code)

    # Data-driven argument mapping
    ARG_TO_CONFIG_MAP = {
        'duration': 'evaluation_duration',
        'elite_rate': 'elite_selection_rate',
        'crossover_rate': 'crossover_rate',
        'mutation_rate': 'mutation_rate',
        'power_min': 'power_min',
        'power_max': 'power_max',
        'max_filters': 'max_filters',
        'min_hits_min': 'min_hits_min',
        'min_hits_max': 'min_hits_max',
        'adaptation_window': 'adaptation_window',
        'meta_min_rate': 'meta_learning_min_rate',
        'meta_max_rate': 'meta_learning_max_rate',
        'fallback_inf_rate': 'fallback_inf_rate',
        'fallback_finite_rate': 'fallback_finite_rate',
        'mutation_prob_power': 'mutation_prob_power',
        'mutation_prob_filter': 'mutation_prob_filter',
        'mutation_prob_modulus': 'mutation_prob_modulus',
        'mutation_prob_residue': 'mutation_prob_residue',
        'mutation_prob_add_filter': 'mutation_prob_add_filter',
    }

    # Build overrides dict from args
    overrides = {}
    for arg_name, config_name in ARG_TO_CONFIG_MAP.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            overrides[config_name] = arg_value

    # Handle special cases (prometheus flags)
    if hasattr(args, "prometheus") and args.prometheus:
        overrides["prometheus_enabled"] = True
    if hasattr(args, "prometheus_mode") and args.prometheus_mode is not None:
        overrides["prometheus_mode"] = args.prometheus_mode
    if hasattr(args, "max_api_cost") and args.max_api_cost is not None:
        overrides["max_api_cost"] = args.max_api_cost

    return cls(api_key=api_key, **{**base_dict, **overrides})
```

**Benefits**:
- Reduces 87 lines to ~35 lines (60% reduction)
- Eliminates copy-paste errors
- Self-documenting CLI → Config mapping
- Easy to add new parameters

---

### 4. MEDIUM: Long Functions

**Impact**: Medium - Reduces readability and testability
**Scope**: 8 functions exceed 80 lines

#### Functions Exceeding Length Threshold:

| Function | Lines | File | Complexity |
|----------|-------|------|------------|
| `run_collaborative_evolution()` | 136 | `prometheus/experiment.py` | High |
| `run_independent_baseline()` | 117 | `prometheus/experiment.py` | High |
| `process_request()` (SearchSpecialist) | 70 | `prometheus/agents.py` | Medium |
| `_build_feedback_prompt()` | 44 | `llm/gemini.py` | Low (string) |
| `_build_prompt()` | 60 | `llm/gemini.py` | Low (string) |
| `run_evolutionary_cycle()` | 224 | `evolution.py` | High |
| `from_args_and_env()` | 78 | `config.py` | Low |
| `_enforce_rate_bounds()` | 47 | `adaptive_engine.py` | Medium |

#### Recommendation:
**Action**: Extract sub-methods for complex functions
**Priority**: MEDIUM
**Effort**: High (6-8 hours for all)

**Example - Refactor `run_evolutionary_cycle()`**:

Current structure (224 lines):
```python
def run_evolutionary_cycle() -> tuple[float, Strategy]:
    # 1. Evaluation (40 lines)
    # 2. Selection (10 lines)
    # 3. Meta-learning updates (30 lines)
    # 4. Meta-learning adaptation (15 lines)
    # 5. Reproduction (85 lines)
    # 6. Return (5 lines)
```

Proposed structure:
```python
def run_evolutionary_cycle() -> tuple[float, Strategy]:
    """Run one generation: evaluate, select, reproduce."""
    generation_metrics = self._evaluate_population()
    elites = self._select_elites()
    best_fitness, best_strategy = elites[0][1]["fitness"], elites[0][1]["strategy"]

    self._update_fitness_history(best_fitness)

    if self.meta_learner:
        self._update_meta_learning(elites)
        self._adapt_rates_if_ready()

    self.civilizations = self._create_next_generation(elites)
    self.generation += 1

    return best_fitness, best_strategy

def _evaluate_population(self) -> List[EvaluationMetrics]:
    """Evaluate all civilizations and log results."""
    # ... (40 lines extracted)

def _select_elites(self) -> List[Tuple[str, Dict]]:
    """Select top performers for reproduction."""
    # ... (10 lines extracted)

def _update_meta_learning(self, elites: List) -> None:
    """Update operator statistics based on elite selection."""
    # ... (30 lines extracted)

def _adapt_rates_if_ready(self) -> None:
    """Adapt operator rates if past adaptation window."""
    # ... (15 lines extracted)

def _create_next_generation(self, elites: List) -> Dict[str, Dict]:
    """Generate offspring using crossover, mutation, and random."""
    # ... (85 lines extracted)
```

**Benefits**:
- Main method becomes 20 lines (89% reduction)
- Each sub-method is independently testable
- Clear separation of concerns
- Easier to understand control flow

---

### 5. MEDIUM: Embedded Prompt Strings

**Impact**: Medium - Reduces testability, makes prompt iteration harder
**Scope**: `src/llm/gemini.py` - Two large prompt strings

#### Problem:
Prompts are embedded as f-strings within methods (`_build_prompt()` and `_build_feedback_prompt()`), making them:
- Hard to version control separately
- Difficult to A/B test
- Impossible to load from external files
- Hard to review in diffs

#### Recommendation:
**Action**: Extract prompts to template files or constants
**Priority**: MEDIUM
**Effort**: Low (1-2 hours)

**Option A: Template Files** (Preferred)
```
src/llm/prompts/
  ├── mutation_base.txt
  ├── mutation_feedback.txt
  └── __init__.py
```

```python
# src/llm/prompts/__init__.py
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent

def load_prompt(name: str) -> str:
    """Load prompt template by name."""
    return (PROMPTS_DIR / f"{name}.txt").read_text()

MUTATION_BASE_TEMPLATE = load_prompt("mutation_base")
MUTATION_FEEDBACK_TEMPLATE = load_prompt("mutation_feedback")
```

```python
# gemini.py
from src.llm.prompts import MUTATION_BASE_TEMPLATE

def _build_prompt(self, parent_strategy, fitness, generation, fitness_history):
    """Build prompt for mutation proposal."""
    return MUTATION_BASE_TEMPLATE.format(
        fitness=fitness,
        generation=generation,
        recent_history=fitness_history[-5:] if fitness_history else [],
        power=parent_strategy["power"],
        modulus_filters=parent_strategy["modulus_filters"],
        # ... etc
    )
```

**Option B: Constants Module** (Simpler)
```python
# src/llm/prompt_templates.py
MUTATION_BASE_TEMPLATE = """
You are optimizing heuristics for the General Number Field Sieve (GNFS).

## Task
Propose a mutation to improve a strategy...
"""

MUTATION_FEEDBACK_TEMPLATE = """
You are a SearchSpecialist in an AI civilization...
"""
```

**Benefits**:
- Prompts are versionable assets
- Easier to A/B test prompt variations
- Non-engineers can review/edit prompts
- Cleaner diffs for prompt changes
- Can load different prompts based on config

---

### 6. LOW: Type Hint Gaps

**Impact**: Low - Minor reduction in IDE support and type safety
**Scope**: Several functions missing complete annotations

#### Examples:
```python
# evolution.py:27-36
def __init__(
    self,
    crucible: FactorizationCrucible,
    population_size: int = 10,
    config: Optional[Config] = None,
    llm_provider=None,  # ❌ Missing type hint
    random_seed: Optional[int] = None,
    enable_meta_learning: bool = False,
    generator: Optional["StrategyGenerator"] = None,
):

# comparison.py:144
llm_provider=None,  # ❌ Missing type hint
```

#### Recommendation:
**Action**: Add missing type hints
**Priority**: LOW
**Effort**: Low (1 hour)

```python
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.base import LLMProvider

def __init__(
    self,
    crucible: FactorizationCrucible,
    population_size: int = 10,
    config: Optional[Config] = None,
    llm_provider: Optional["LLMProvider"] = None,  # ✅ Fixed
    random_seed: Optional[int] = None,
    enable_meta_learning: bool = False,
    generator: Optional["StrategyGenerator"] = None,
):
```

---

## Additional Observations

### Strengths

1. **Excellent Test Coverage** - 339 tests with comprehensive edge case coverage
2. **Good Documentation** - Detailed docstrings and CLAUDE.md reference
3. **Modular Architecture** - Clear separation of concerns (strategy, evolution, LLM, prometheus)
4. **Configuration Management** - Centralized Config dataclass with validation
5. **Error Handling** - Defensive coding patterns (try/except, warnings, fallbacks)
6. **Pre-commit Hooks** - Automated quality checks (ruff, pytest)
7. **Meta-Learning System** - Sophisticated UCB1-based operator adaptation

### Weaknesses

1. **Language Mixing** - Japanese comments violate internationalization best practices
2. **Code Duplication** - Baseline evaluation logic repeated 3 times
3. **Long Functions** - Several functions exceed 100 lines
4. **Embedded Strings** - Prompts mixed with code logic
5. **Configuration Verbosity** - 87-line if-statement chain

---

## Refactoring Priority Matrix

| Priority | Issue | Effort | Impact | Risk |
|----------|-------|--------|--------|------|
| 1 | Japanese Comments → English | High | High | Low |
| 2 | Code Duplication (baselines) | Medium | High | Low |
| 3 | Config Verbosity | Low | Medium | Low |
| 4 | Long Functions (evolution.py) | High | Medium | Medium |
| 5 | Embedded Prompts → Templates | Low | Medium | Low |
| 6 | Type Hints | Low | Low | Low |

---

## Recommended Refactoring Sequence

### Phase 1: Critical Cleanup (Week 1)
**Goal**: Address language and duplication issues

1. **Translate Japanese Comments** (Day 1-2)
   - Use automated translation for first pass
   - Manual review by bilingual developer
   - Run full test suite to verify no breakage
   - Update style guide in CLAUDE.md

2. **Refactor Baseline Duplication** (Day 3-4)
   - Extract `_run_baseline_with_strategy_source()` helper
   - Simplify `run_independent_baseline()`
   - Add tests for new helper method
   - Verify all baselines produce same results

3. **Config Verbosity** (Day 5)
   - Implement data-driven ARG_TO_CONFIG_MAP
   - Simplify `from_args_and_env()`
   - Update tests if needed

### Phase 2: Structural Improvements (Week 2)
**Goal**: Improve maintainability

4. **Extract Prompts** (Day 1)
   - Create `src/llm/prompts/` directory
   - Move prompts to template files
   - Update GeminiProvider to use templates
   - Add prompt versioning

5. **Refactor Long Functions** (Day 2-4)
   - Start with `run_evolutionary_cycle()` (highest impact)
   - Extract sub-methods with clear responsibilities
   - Add unit tests for extracted methods
   - Refactor `run_collaborative_evolution()` next

6. **Type Hints** (Day 5)
   - Add missing type annotations
   - Run mypy to verify
   - Update tests if needed

### Phase 3: Testing & Documentation (Week 3)
**Goal**: Ensure quality and capture learnings

7. **Integration Testing** (Day 1-2)
   - Run full test suite (339 tests)
   - Add regression tests for refactored code
   - Performance benchmarking (ensure no slowdown)

8. **Documentation Updates** (Day 3-4)
   - Update CLAUDE.md with new patterns
   - Document new helper methods
   - Update architecture diagrams if needed

9. **Code Review & Merge** (Day 5)
   - Create PR with detailed changelog
   - Address reviewer feedback
   - Merge to main

---

## Risk Mitigation

### Testing Strategy
- Run full test suite after EACH refactoring step
- Use `git bisect` if regressions appear
- Compare metrics exports before/after to verify identical behavior

### Rollback Plan
- Each phase is in separate PR
- Can revert individual PRs if issues arise
- Tag codebase before Phase 1 as `pre-refactor-baseline`

### Success Metrics
- All 339 tests passing
- No performance regression (< 5% slowdown)
- Code coverage maintained (> 95%)
- Reduced lines of code (target: -20%)
- No new mypy/ruff warnings

---

## Implementation Checklist

### Phase 1: Critical Cleanup
- [ ] Create feature branch: `refactor/phase-1-cleanup`
- [ ] Translate Japanese comments to English
- [ ] Run full test suite (339 tests)
- [ ] Extract baseline evaluation helper
- [ ] Simplify Config.from_args_and_env()
- [ ] Create PR #1 with detailed changelog
- [ ] Review and merge

### Phase 2: Structural Improvements
- [ ] Create feature branch: `refactor/phase-2-structure`
- [ ] Extract prompt templates
- [ ] Refactor run_evolutionary_cycle()
- [ ] Refactor run_collaborative_evolution()
- [ ] Add missing type hints
- [ ] Run mypy validation
- [ ] Create PR #2 with before/after metrics
- [ ] Review and merge

### Phase 3: Testing & Documentation
- [ ] Run full integration test suite
- [ ] Performance benchmarking
- [ ] Update CLAUDE.md
- [ ] Update architecture docs
- [ ] Create final summary PR #3
- [ ] Tag release: `v2.0-refactored`

---

## Conclusion

The Factorization codebase is well-architected with strong testing and documentation practices. However, the extensive use of Japanese comments creates a critical barrier to international collaboration and long-term maintenance.

**Immediate Action Required**:
1. Translate all Japanese comments to English (CRITICAL)
2. Eliminate code duplication in baseline evaluation (HIGH)
3. Simplify configuration verbosity (HIGH)

**Timeline**: 3 weeks for complete refactoring
**Risk Level**: Low - Excellent test coverage provides safety net
**Expected Benefits**:
- 20% reduction in LOC
- Improved international accessibility
- Better maintainability
- Enhanced testability

**Recommendation**: Proceed with Phase 1 immediately. The codebase quality is good, but these improvements will make it excellent.
