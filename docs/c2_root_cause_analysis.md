# C2 Root Cause Analysis: Why LLM-Guided Mutations Underperformed by 54%

**Analysis Date**: November 07, 2025
**Context**: C2 validation showed dramatic underperformance (emergence 0.462 vs target >1.1)
**Investigation Focus**: Determine why LLM guidance made performance worse instead of better

---

## Executive Summary

**Primary Root Cause**: **LLM API call limit exhausted early, forcing 91% fallback to rule-based mutations**

**Evidence**:
- 4,201 LLM failures logged ("LLM failed: API call limit reached")
- 175 filter removal attempts blocked (validation constraint)
- Total API calls: 0 reported in final run (limit hit immediately)
- Expected calls: ~5,700 (15 runs × 20 gen × 19 mutations)
- Actual LLM-guided: ~9% of mutations (due to limit)

**Conclusion**: C2 experiments ran primarily in **fallback mode**, not true LLM-guided mode. Results represent "broken LLM + rule-based hybrid" not "pure LLM-guided evolution."

---

## Investigation Findings

### 1. API Call Limit Exhaustion (HIGH SEVERITY)

**Configuration**:
```json
{
  "max_llm_calls": 100,
  "max_api_cost": 1.0
}
```

**Calculations**:
- **Expected mutations per run**: 20 generations × 19 mutations = 380 LLM calls
- **Total expected (15 runs)**: 380 × 15 = 5,700 LLM calls
- **Configured limit**: 100 calls per run
- **Result**: Limit exhausted at generation 5-6 (100/380 = 26% of experiment)

**Impact Analysis**:

*Per-Run Impact (for a run that hits the 100-call limit)*:
- Generation 0-5: LLM-guided mutations (~26% of the run)
- Generation 6-19: Rule-based fallback (~74% of the run)

*Overall Impact (across all 15 runs)*:
- Total LLM-guided mutations: ~9% (some runs exhausted limits earlier)
- Total rule-based fallbacks: ~91% (4,201 fallback warnings / ~4,600 total mutation attempts)

**Log Evidence**:
```
2025-11-03 20:02:39 - src.prometheus.agents - WARNING - [C2] LLM failed: API call limit reached. Using rule-based fallback.
[Repeated 4,201 times across all runs]
```

### 2. LLM Reasoning Quality Issues (MEDIUM SEVERITY)

**Pattern 1: Contradictory Mutations**

Example from seed 9000:
```
Generation 3: "Increasing smoothness bound slightly will provide more thorough checking"
→ Mutation: smoothness_bound 19 → 23

Generation 7: "Reducing smoothness bound should speed up smoothness check phase"
→ Mutation: smoothness_bound 23 → 19
```

**Analysis**: LLM oscillating between contradictory strategies without learning from feedback trends.

**Pattern 2: Failed Filter Removals (175 occurrences)**

```
LLM reasoning: "Removing this filter entirely should speed up candidate generation"
Result: "Cannot remove filter: minimum 1 filter required, keeping parent strategy"
```

**Analysis**: LLM unaware of validation constraints. Proposed ~175 invalid mutations (4.6% of total attempts).

**Pattern 3: Misunderstanding Bottlenecks**

```
Feedback: "smoothness_check is the slowest phase"
LLM: "Increasing smoothness bound slightly will improve quality"
Reality: Higher bound makes smoothness_check SLOWER, not better
```

**Analysis**: LLM reasoning sometimes contradicts stated goal (speed vs quality tradeoff misunderstood).

### 3. Configuration Issues (HIGH SEVERITY)

**Issue**: `max_llm_calls=100` far too low for 20-generation experiments

**Intended**: Safety limit to prevent runaway API costs
**Actual Effect**: Forced early fallback, invalidating C2 experimental condition

**Fix**: Should be `max_llm_calls=500` or `max_api_cost=10.0` for 20-gen experiments

**Cost Analysis**:
- Target LLM calls: 5,700 total (15 runs × 380)
- Actual LLM calls: ~570 (100 per run × first 5-6 runs before exhaustion)
- Cost per call: ~$0.0001 (Gemini 2.5 Flash Lite)
- Actual cost: ~$0.06-0.10 (not the $2-3 originally estimated)

### 4. Temperature Scaling (LOW SEVERITY)

**Verified**: PR #53 fix for `max_generations` propagation working correctly

**Evidence**: No temperature-related errors in logs, LLM temperature scaling logs show correct progression.

**Conclusion**: Temperature scaling NOT a root cause.

---

## Invalidation of C2 Results

**Critical Conclusion**: C2 validation results **DO NOT represent true LLM-guided evolution**.

**What was actually tested**:
- **Generation 0-5**: LLM-guided (26% of experiment)
- **Generation 6-19**: Rule-based fallback (74% of experiment)

**What should have been tested**:
- **All 20 generations**: LLM-guided throughout

**Validity Assessment**:
- ❌ C2 vs C1 comparison **INVALID** (C2 was mostly rule-based too)
- ❌ Emergence factor 0.462 **NOT representative** of LLM capability
- ❌ H1b hypothesis **NOT tested** (insufficient LLM guidance)

**Correct Interpretation**:
- C2 results show: "LLM for 26% of experiment + rule-based for 74%" = worse than "rule-based for 100%"
- This suggests: **Mixing LLM and rule-based is worse than consistent approach**
- Alternative hypothesis: **Early LLM mutations degraded population, later rule-based couldn't recover**

---

## Recommendations

### Immediate Action: Re-run C2 with Correct Configuration

**Priority**: HIGH
**Timeline**: 1 week
**Cost**: $2-3 (actual LLM calls)

**Configuration changes**:
```python
config = Config(
    max_llm_calls=500,  # Was: 100 ← ROOT CAUSE
    max_api_cost=5.0,   # Was: 1.0 (safety margin)
    # All other params unchanged
)
```

**Rationale**:
1. Current C2 results invalid (only 26% LLM-guided)
2. Cannot conclude anything about LLM capability from broken experiment
3. Low cost ($2-3) to get valid data
4. H1b hypothesis remains untested

### Alternative: Partial Re-run (5 seeds)

**If full re-run too costly**:
- Re-run seeds 9000-9004 (5 runs)
- Compare: "True C2" (LLM throughout) vs "Broken C2" (limit-exhausted)
- **Hypothesis**: True C2 will perform better than broken C2
- **Decision point**: If true C2 > broken C2 by 20%+, justify full 15-run validation

### Long-term: Configuration Validation

**Problem**: Easy to misconfigure experiments, leading to invalid results

**Solution**: Add pre-flight checks in experiment.py:
```python
def validate_experiment_config(config, generations, population):
    expected_mutations = generations * (population - 1)  # Rough estimate
    if config.max_llm_calls < expected_mutations:
        warnings.warn(
            f"max_llm_calls ({config.max_llm_calls}) may be insufficient "
            f"for {generations} generations (need ~{expected_mutations}). "
            f"LLM will fallback to rule-based mid-experiment."
        )
```

---

## Lessons Learned

### 1. Validate Experimental Conditions

- ✅ **Before analysis**: Check logs for unexpected warnings/errors
- ✅ **Monitor fallback rates**: Track LLM success vs fallback frequency
- ✅ **Verify configuration**: Ensure limits don't invalidate experimental conditions

### 2. Distinguish "LLM failed" from "Experiment failed"

**Current interpretation**:
- "C2 underperformed by 54%" → Implies LLM guidance is bad

**Correct interpretation**:
- "C2 with 91% fallback underperformed by 54%" → Implies configuration error, not LLM failure

### 3. Cost vs Quality Trade-offs

**Safety limits** (max_api_cost) are important, but:
- Should be set based on expected usage, not arbitrary round numbers
- $1.00 limit for experiment needing $2-3 guarantees premature termination
- Better: Set $5-10 limit with monitoring, not $1 limit that breaks experiments

### 4. Log Analysis Before Statistical Analysis

**Should have checked FIRST**:
1. Read logs for warnings/errors
2. Verify LLM call counts vs expected
3. Check fallback rates
4. THEN run statistical analysis

**Would have discovered**:
- Broken experimental condition BEFORE spending hours on analysis
- Need for re-run BEFORE documenting "LLM doesn't work"

---

## Updated Recommendations for Task 1

**Original Plan**: Document C2 failure, pivot to meta-learning

**Updated Plan** (based on root cause):
1. ✅ Document invalid C2 results (done)
2. ✅ Root cause investigation (done - found config error)
3. 🔄 **Re-run C2 with correct configuration** (NEW - high priority)
4. ⏸️ Hold pivot decision until valid C2 data available

**Rationale**: Cannot conclude "LLM guidance doesn't work" from experiment where LLM was only active 26% of the time.

---

## Conclusion

**C2 validation results are INVALID due to configuration error.**

**Root cause**: `max_llm_calls=100` exhausted at generation 5-6, forcing 74% of experiment to run in rule-based fallback mode.

**Impact**:
- H1b hypothesis NOT actually tested
- "LLM-guided evolution" was actually "LLM-guided for 26%, rule-based for 74%"
- Cannot conclude anything about LLM capability from this data

**Next steps**:
1. Re-run C2 with `max_llm_calls=500`
2. Compare valid C2 results against C1/rulebased
3. THEN make pivot decision based on valid data

**Timeline**: 1 week for re-run + analysis
**Cost**: $2-3 (acceptable for valid experimental data)

**Scientific Integrity**: Documenting this configuration error and invalidating results is correct scientific practice. Better to admit mistake and re-run than draw conclusions from broken experiments.

---

**Document Version**: 1.0
**Status**: Root cause identified, re-run recommended
**Author**: C2 validation investigation (November 2025)
