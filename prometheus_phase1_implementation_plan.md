# Prometheus Phase 1 MVP: Detailed Implementation Plan

**Date**: November 01, 2025
**Timeline**: 2 weeks (80 hours)
**Budget**: <$1.21
**Goal**: Test H1: Collaboration > Independence

---

## Executive Summary

Implement dual-agent system where SearchSpecialist and EvaluationSpecialist collaborate to evolve better strategies than independent agents.

**Success Criteria**:
- p-value < 0.05 (statistical significance)
- Cohen's d ≥ 0.5 (medium effect size)
- Emergence factor > 1.1 (10%+ improvement)

---

## Work Breakdown Structure

### Week 1: Core Implementation (40 hours)

#### Foundation (8 hours)
- **Task 1.1**: Design agent interfaces (2h)
- **Task 1.2**: Design communication protocol (2h)
- **Task 1.3**: Create `src/prometheus/` module structure (1h)
- **Task 1.4**: Update Config for Prometheus (3h)

#### Agent Implementation (12 hours)
- **Task 2.1**: Implement `CognitiveCell` base class (3h)
- **Task 2.2**: Implement `SearchSpecialist` (4h)
- **Task 2.3**: Implement `EvaluationSpecialist` (4h)
- **Task 2.4**: Unit tests for agents (1h)

#### Communication Layer (8 hours)
- **Task 3.1**: Implement `SimpleCommunicationChannel` (3h)
- **Task 3.2**: Implement message routing (2h)
- **Task 3.3**: Integration tests for communication (3h)

#### Experiment Framework (12 hours)
- **Task 4.1**: Implement `PrometheusExperiment` orchestrator (4h)
- **Task 4.2**: Implement baseline experiment runner (2h)
- **Task 4.3**: Implement emergence metrics calculation (3h)
- **Task 4.4**: End-to-end tests (3h)

### Week 2: Testing, Experiments, Analysis (40 hours)

#### Testing & Validation (12 hours)
- **Task 5.1**: Expand unit test coverage (4h)
- **Task 5.2**: Integration test suite (4h)
- **Task 5.3**: Local CI validation (2h)
- **Task 5.4**: Mock LLM testing (2h)

#### Experiment Execution (16 hours)
- **Task 6.1**: Pilot experiment (4h)
- **Task 6.2**: Baseline experiment (4h)
- **Task 6.3**: Collaborative experiment (6h)
- **Task 6.4**: Data validation (2h)

#### Analysis & Reporting (12 hours)
- **Task 7.1**: Statistical analysis (4h)
- **Task 7.2**: Visualization (4h)
- **Task 7.3**: Documentation (3h)
- **Task 7.4**: Go/No-Go decision (1h)

---

## Component Architecture

### CognitiveCell (Base Agent Class)

```python
@dataclass
class Message:
    sender_id: str
    recipient_id: str
    message_type: str  # "strategy_request", "evaluation_request", "feedback"
    payload: dict
    timestamp: float
    conversation_id: Optional[str] = None

class CognitiveCell(ABC):
    """Abstract base class for specialized agents."""

    def __init__(self, agent_id: str, config: Config):
        self.agent_id = agent_id
        self.config = config
        self.memory = AgentMemory()

    @abstractmethod
    def process_request(self, message: Message) -> Message:
        """Process incoming request and generate response."""
        pass
```

### SearchSpecialist

Generates novel strategies using LLM or rule-based mutations, incorporating feedback from EvaluationSpecialist.

**Key Methods**:
- `process_request(message)`: Handle feedback, generate new strategy
- `_generate_llm_strategy(feedback)`: Use Gemini with feedback context
- `_build_strategy_prompt(feedback)`: Create prompt incorporating feedback

### EvaluationSpecialist

Evaluates strategies using FactorizationCrucible and provides actionable feedback.

**Key Methods**:
- `process_request(message)`: Evaluate strategy, generate feedback
- `_generate_feedback(strategy, metrics)`: Analyze metrics, create feedback
- `_analyze_bottlenecks(metrics)`: Identify performance issues

### SimpleCommunicationChannel

Synchronous request/response communication with logging.

**Key Methods**:
- `register_agent(agent)`: Register agent to receive messages
- `send_message(message)`: Send message, return response
- `get_communication_stats()`: Return message counts, types

### PrometheusExperiment

Orchestrates dual-agent evolution and baseline comparisons.

**Key Methods**:
- `run_collaborative_evolution(generations, population_size)`: Main experiment
- `run_independent_baseline(agent_type, generations, population_size)`: Single-agent baseline
- `compare_with_baselines(generations, population_size)`: Full comparison

---

## Experiment Protocol

### Pilot Experiment (Verify Infrastructure)

```bash
# Rule-based pilot (no API cost)
python main.py --prometheus --prometheus-mode collaborative \
  --generations 10 --population 10 --duration 0.5 --seed 1000 \
  --export-metrics results/prometheus_pilot_rulebased.json

# LLM pilot (small API cost ~$0.01)
python main.py --prometheus --prometheus-mode collaborative --llm \
  --generations 5 --population 5 --duration 0.5 --seed 2000 \
  --export-metrics results/prometheus_pilot_llm.json
```

### Baseline Experiments (30 runs)

```bash
# Search-only baseline
for seed in {1000..1009}; do
  python main.py --prometheus --prometheus-mode search_only \
    --generations 20 --population 20 --duration 1.0 --seed $seed \
    --export-metrics results/baseline_search_${seed}.json
done

# Eval-only baseline
for seed in {1000..1009}; do
  python main.py --prometheus --prometheus-mode eval_only \
    --generations 20 --population 20 --duration 1.0 --seed $seed \
    --export-metrics results/baseline_eval_${seed}.json
done

# Rule-based baseline
for seed in {1000..1009}; do
  python main.py --generations 20 --population 20 --duration 1.0 --seed $seed \
    --export-metrics results/baseline_rulebased_${seed}.json
done
```

### Collaborative Experiments (10 runs with LLM)

```bash
for seed in {2000..2009}; do
  python main.py --prometheus --prometheus-mode collaborative --llm \
    --generations 20 --population 20 --duration 1.0 --seed $seed \
    --export-metrics results/collaborative_${seed}.json
done
```

**API Cost**: ~$0.80 (2000 messages × $0.0004/message)

---

## Statistical Analysis

### Hypothesis Testing

```python
from src.statistics import StatisticalAnalyzer

# Load aggregated data
analyzer = StatisticalAnalyzer()
result = analyzer.compare_fitness_distributions(
    evolved_scores=collaborative_fitness_values,
    baseline_scores=best_baseline_fitness_values
)

# Check success criteria
h1_confirmed = (
    result.p_value < 0.05 and
    result.effect_size >= 0.5 and
    emergence_factor > 1.1
)
```

### Emergence Metrics

```python
emergence_factor = collaborative_fitness / max(baseline_fitnesses)
synergy_score = collaborative_fitness - max(baseline_fitnesses)
communication_efficiency = fitness_gain / total_messages
```

---

## Success Criteria

### Code Deliverables

**New Files**:
- [ ] `src/prometheus/__init__.py`
- [ ] `src/prometheus/agents.py`
- [ ] `src/prometheus/communication.py`
- [ ] `src/prometheus/experiment.py`
- [ ] `tests/prometheus/test_agents.py` (50+ tests)
- [ ] `tests/prometheus/test_communication.py` (20+ tests)
- [ ] `tests/prometheus/test_experiment.py` (30+ tests)

**Code Quality**:
- [ ] Test coverage ≥90% for `src/prometheus/`
- [ ] All tests pass `make ci`
- [ ] No type errors, lint warnings

### Experiment Deliverables

**Data Files** (43 total):
- [ ] 2 pilot files
- [ ] 30 baseline files (10 each × 3 baselines)
- [ ] 10 collaborative files
- [ ] 1 aggregated results file

### Analysis Deliverables

**Statistical Results**:
- [ ] Welch's t-test results (p-values)
- [ ] Cohen's d effect sizes
- [ ] Emergence factor distribution
- [ ] Communication efficiency metrics

**Visualizations**:
- [ ] Fitness trajectories over generations
- [ ] Final fitness distributions (box plots)
- [ ] Emergence factor histogram
- [ ] Communication pattern analysis

### Documentation Deliverables

- [ ] `docs/prometheus_phase1_results.md` (Results summary)
- [ ] Update `README.md` (Prometheus usage)
- [ ] Update CLI help (new flags)
- [ ] Code docstrings (all public APIs)

---

## Go/No-Go Decision Framework

| Scenario | p-value | Cohen's d | Decision |
|----------|---------|-----------|----------|
| Strong Success | <0.01 | ≥0.8 | **GO to Phase 2** |
| Moderate Success | <0.05 | ≥0.5 | **GO to Phase 2** |
| Weak Signal | <0.1 | ≥0.3 | **CONDITIONAL GO** (extend experiments) |
| No Signal | ≥0.1 | <0.3 | **NO-GO** (analyze failure) |
| Negative | N/A | <0 | **NO-GO** (redesign) |

---

## Risk Mitigation

### Risk 1: API Costs Exceed Budget

**Mitigation**:
- Run rule-based pilot first
- Monitor costs with `GeminiProvider._call_count`
- Hard limit: `max_api_cost = 0.50` in config

### Risk 2: Agents Don't Collaborate

**Early Detection**:
- Monitor `communication_stats`
- Check if feedback is actionable
- Verify SearchSpecialist uses feedback

**Fallback**:
- Improve prompt engineering
- Add explicit feedback citation requirement

### Risk 3: Insufficient Statistical Power

**Mitigation**:
- Run power analysis after pilot
- Increase n if needed (budget allows n=20)

---

## Timeline

**Week 1**: Implementation + Testing
- Mon-Tue: Foundation + Agents
- Wed: Communication
- Thu-Fri: Experiment framework + Testing

**Week 2**: Experiments + Analysis
- Mon: Pilot experiments
- Tue-Wed: Full experiments (baseline + collaborative)
- Thu: Statistical analysis
- Fri: Documentation + Go/No-Go decision

---

## Cost Breakdown

| Component | API Calls | Cost |
|-----------|-----------|------|
| Pilot (LLM) | 50 | $0.03 |
| Search-only baseline | 4000 | $0.24 |
| Collaborative | 4000 | $0.24 |
| **Total** | **~8050** | **~$0.51** |

**Note**: Eval-only and rule-based baselines have $0 cost (no LLM calls).

**Contingency**: If costs exceed $0.50, reduce collaborative runs from 10 to 8.

---

## Quick Reference Commands

### Development
```bash
# Run tests
pytest tests/prometheus/ -v

# Type checking
mypy src/prometheus/

# Formatting
make format

# Full CI
make ci
```

### Experiments
```bash
# Pilot
python main.py --prometheus --prometheus-mode collaborative \
  --generations 10 --population 10 --seed 1000

# Full experiment
python main.py --prometheus --prometheus-mode collaborative --llm \
  --generations 20 --population 20 --seed 2000 \
  --export-metrics results/collaborative_2000.json
```

### Analysis
```python
# Statistical test
from src.statistics import StatisticalAnalyzer
analyzer = StatisticalAnalyzer()
result = analyzer.compare_fitness_distributions(
    evolved_scores=collaborative,
    baseline_scores=baselines
)
print(f"p={result.p_value:.4f}, d={result.effect_size:.2f}")
```

---

## Definition of Done

Phase 1 MVP is complete when:

1. ✅ All code committed and passing CI
2. ✅ 100+ tests written and passing (90%+ coverage)
3. ✅ 43 experiment runs completed successfully
4. ✅ Statistical analysis shows H1 result (confirmed or not)
5. ✅ Results documented with visualizations
6. ✅ Go/No-Go decision made for Phase 2
7. ✅ User validation: outputs correct, no timeouts, proper formatting
8. ✅ README and CLI help updated

---

**Ready for immediate execution** - Follow this plan step-by-step with TDD principles.
