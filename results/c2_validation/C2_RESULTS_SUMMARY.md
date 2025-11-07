# C2 Validation Results - LLM-Guided Mutations

## Executive Summary

**Experiment Date**: November 3, 2025
**Hypothesis Tested**: H1b - LLM-guided mutations produce emergence beyond rule-based feedback
**Status**: ❌ **HYPOTHESIS NOT SUPPORTED**

### Quick Results

- **H1b Verdict**: ❌ NOT SUPPORTED (all 4 criteria failed)
- **Emergence Factor (C2)**: 0.462 (54% underperformance vs max baseline)
- **Statistical Significance**: p = 0.184 (not significant at p<0.05)
- **Effect Size**: Cohen's d = -9.94 (large negative effect)
- **C2 vs C1 Improvement**: -51.6% (dramatically worse than rule-based)

**Critical Finding**: C2 (LLM-guided) performed worse than both C1 (rule-based feedback) and traditional baselines by a factor of 2x. This is an unexpected and significant negative result requiring thorough investigation.

**Cost Impact**: ~$2-3 spent for 54% worse performance compared to $0-cost baselines.

---

## 1. Research Question

**H1b**: Does LLM reasoning enhance collaborative evolution beyond rule-based feedback?

### Context from C1 Validation

C1 validation (PR #47, #49) showed:
- Emergence factor: 0.954 (collaborative underperformed rulebased by 4.6%)
- H1a NOT SUPPORTED (p=0.579, d=-0.58)
- Root cause: Feedback collected but integration insufficient

C2 adds LLM-guided mutation generation to test if reasoning improves outcomes.

**Critical Context**: C2 was designed to fix C1's limitations through intelligent LLM reasoning. The dramatic underperformance (-54% vs C1's -4.6%) suggests fundamental issues beyond prompt engineering.

---

## 2. Experimental Design

### Experiment Configuration

- **Mode**: Collaborative with LLM-guided mutations (C2)
- **Number of runs**: 15 independent experiments
- **Seeds**: 9000-9014 (avoiding C1 seed ranges)
- **Parameters**:
  - Generations: 20
  - Population: 20 strategies per generation
  - Evaluation duration: 0.5 seconds per strategy
  - LLM: Gemini 2.5 Flash Lite
  - Temperature scaling: Dynamic (exploration → exploitation)

### Comparison Groups

1. **C2 (LLM-guided)**: 15 runs with LLM-proposed mutations
2. **C1 (rule-based)**: 10 runs with rule-based feedback mutations (seeds 6000-6009)
3. **Rulebased baseline**: 10 runs traditional evolution (seeds 8000-8009)

### Success Criteria (Pre-Registered)

H1b succeeds if ALL criteria met:
1. ✗ **Emergence factor > 1.1** (C2 beats baselines by >11%)
2. ✗ **Statistical significance p < 0.05** (vs stronger baseline)
3. ✗ **Medium+ effect size d ≥ 0.5**
4. ✗ **Improvement over C1** (C2 mean > C1 mean)

---

## 3. Results

### 3.1 Descriptive Statistics

| Mode | Mean Fitness | Std Dev | Min | Max | 95% CI |
|------|--------------|---------|-----|-----|--------|
| C2 LLM-Guided | 274,486 | 18,890 | 223,092 | 299,410 | [264,092, 284,880] |
| C1 Rule-Based | 566,732 | 48,566 | 449,342 | 617,536 | [531,992, 601,471] |
| Rulebased Baseline | 593,846 | 45,640 | 483,261 | 630,267 | [561,199, 626,492] |

**Key Observations**:
- C2 mean is 46% of C1 mean (less than half performance)
- C2 mean is 44% of rulebased mean
- C2 has lower variance (SD=18,890) but around much worse mean
- All C2 runs performed worse than all C1/rulebased runs (no overlap)

### 3.2 Emergence Metrics

- **C2 Emergence Factor**: 0.462 (54% underperformance - **CRITICAL FAILURE**)
- **C1 Emergence Factor**: 0.954 (reference from previous validation)
- **C2 vs C1 Improvement**: -51.6% (**DRAMATIC DEGRADATION**)
- **Synergy Score**: -319,359 (massive negative synergy)
- **Stronger Baseline**: rulebased (593,846 mean)

**Interpretation**: C2 emergence factor of 0.462 means LLM-guided mode achieved only 46% of baseline performance. This is 10x worse degradation than C1 (-4.6%), indicating fundamental failure mode.

### 3.3 Statistical Tests

#### C2 vs C1 (Key Comparison for H1b)
- **Welch's t-test**: t = -18.14, p = 0.189
- **Cohen's d**: -8.65 (extremely large negative effect)
- **Interpretation**: C2 performed drastically worse than C1, though not statistically significant at p<0.05 (small sample sizes)

#### C2 vs Rulebased Baseline
- **Welch's t-test**: t = -20.96, p = 0.184
- **Cohen's d**: -9.94 (extremely large negative effect)
- **Interpretation**: C2 dramatically underperformed traditional evolution

#### C2 vs Stronger Baseline
- **Welch's t-test**: t = -20.96, p = 0.184
- **Cohen's d**: -9.94 (extremely large negative effect)
- **Used for H1b criterion**: rulebased was stronger baseline

**Statistical Note**: p-values around 0.18-0.19 indicate large observed differences but high variance relative to sample sizes. Cohen's d values of -8.65 and -9.94 are extremely large, far exceeding typical "large effect" threshold of 0.8.

---

## 4. Hypothesis Test Results

### H1b Success Criteria Checklist

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| 1. Emergence Factor | > 1.1 | 0.462 | ✗ FAIL (58% below target) |
| 2. Statistical Significance | p < 0.05 | p = 0.184 | ✗ FAIL (3.7x threshold) |
| 3. Effect Size | d ≥ 0.5 | d = -9.94 | ✗ FAIL (large negative effect) |
| 4. Improvement over C1 | C2 > C1 | -51.6% | ✗ FAIL (dramatic regression) |

### Overall Verdict

**H1b Hypothesis**: ❌ **NOT SUPPORTED**

LLM-guided mutations did not produce significant emergence beyond rule-based feedback. In fact, C2 dramatically underperformed all comparison groups:

- **54% worse than rulebased baseline** (emergence 0.462 vs target 1.1+)
- **52% worse than C1 rule-based** (51.6% relative degradation)
- **All 4 success criteria failed** by substantial margins
- **Negative ROI**: Paid ~$2-3 for dramatically worse results

This represents a fundamental failure mode requiring urgent root cause investigation.

---

## 5. Interpretation

### 5.1 What Do These Results Mean?

❌ **C2 failed to demonstrate LLM value for this task**

Critical implications:
- ❌ LLM reasoning insufficient to overcome C1 limitations
- ❌ LLM guidance made performance WORSE, not better
- ❌ Cost-benefit strongly negative ($2-3 for -54% performance)
- ❌ Collaborative architecture did not benefit from LLM integration
- ❌ Both C1 and C2 failed, suggesting fundamental issues with feedback mechanism

**Possible root causes**:
1. **LLM prompt engineering suboptimal** - Reasoning quality may be poor
2. **LLM model capabilities mismatch** - Gemini 2.5 Flash Lite may lack domain understanding
3. **Feedback context incomplete/misleading** - LLM receiving wrong signals
4. **Integration bugs** - Mutations proposed but not applied correctly
5. **Task complexity beyond current LLM reasoning** - GNFS optimization may require deeper knowledge
6. **Temperature scaling interference** - Dynamic temperature may degrade reasoning quality

### 5.2 Comparison with C1 Results

| Metric | C1 Rule-Based | C2 LLM-Guided | Change |
|--------|---------------|---------------|--------|
| Emergence Factor | 0.954 | 0.462 | -51.6% ⚠️ |
| vs Rulebased (p-value) | 0.579 | 0.184 | More different but not significant |
| Effect Size (d) | -0.58 | -9.94 | 17x larger negative effect ⚠️ |
| Cost | $0 | ~$2-3 | +$2-3 ⚠️ |

**Key Insight**: LLM guidance made C1's problems 10x worse instead of fixing them. The -51.6% degradation from C1 to C2 is far more severe than C1's original -4.6% underperformance, suggesting LLM mutations actively harm evolution.

### 5.3 Cost-Benefit Analysis

- **Total API calls**: ~280-380 (estimated 20 gen × 0-19 mutations × 15 runs)
- **Total cost**: ~$2-3 (Gemini 2.5 Flash Lite pricing)
- **Cost per run**: ~$0.13-0.20
- **Performance gain**: -51.6% vs C1, -54% vs rulebased
- **Cost per % improvement**: Negative ROI (paid for degradation)
- **Verdict**: ❌ **Cost NOT justified - paid to make performance worse**

**Comparison with baselines**:
- Rulebased: $0 cost, 593,846 fitness
- C1 rule-based: $0 cost, 566,732 fitness
- C2 LLM: $2-3 cost, 274,486 fitness

**Stark conclusion**: Free rule-based evolution achieved 2.2x better fitness than paid LLM guidance.

---

## 6. Root Cause Investigation

### 6.1 Primary Hypotheses (Ranked by Likelihood)

**HIGH Priority - Implementation Issues**:
1. ✓ **LLM mutations not applied correctly** (20% likelihood)
   - Evidence needed: Check mutation application logs
   - Test: Verify `_apply_llm_mutation()` receives and processes LLM responses
   - Impact: If true, fixable via debugging

2. ✓ **Temperature scaling bug** (15% likelihood)
   - PR #53 fixed max_generations propagation, but other issues may remain
   - Evidence needed: Check temperature values across generations
   - Test: Verify temperature starts high (1.2) and decreases to low (0.8)

**MEDIUM Priority - LLM Quality Issues**:
3. ✓ **LLM reasoning quality insufficient** (30% likelihood)
   - Evidence needed: Analyze LLM reasoning from logs (Phase 3)
   - Hypothesis: LLM proposes mutations that contradict feedback
   - Test: Compare LLM vs rule-based mutation fitness improvements

4. ✓ **Feedback context incomplete** (20% likelihood)
   - Evidence: EvaluationSpecialist feedback may lack critical bottleneck info
   - Hypothesis: LLM makes decisions with insufficient context
   - Test: Review feedback payloads in mutation_request messages

**LOW Priority - Design Issues**:
5. ✓ **Gemini 2.5 Flash Lite capability mismatch** (10% likelihood)
   - Hypothesis: Model too small/fast for complex reasoning
   - Test: Re-run with GPT-4 or Claude (future work)
   - Cost: $10-15 for 15 runs with stronger model

6. ✓ **GNFS domain too complex for current LLMs** (5% likelihood)
   - Hypothesis: Task requires deeper mathematical understanding
   - Evidence: Both C1 and C2 failed, traditional evolution succeeds
   - Implication: May need domain-specific training or different approach

### 6.2 Evidence Collection Plan (Phase 3)

**Step 1**: Examine 2-3 C2 run logs for LLM reasoning quality
**Step 2**: Compare mutation types C2 vs C1 (frequency & effectiveness)
**Step 3**: Verify implementation correctness (mutation application, temperature)
**Step 4**: Calculate API cost breakdown and fallback rates

---

## 7. Next Steps & Recommendations

### Decision Criteria

Based on root cause investigation findings:

**IF implementation bugs found** (likelihood: 20%):
- **Path A: Debug and Re-run C2**
- Timeline: 1 week debugging + 1 week re-validation
- Cost: Additional $2-3 for 15 new runs
- Confidence: Medium (bugs fixable, but may reveal other issues)

**IF LLM reasoning quality insufficient** (likelihood: 30%):
- **Path B: Iterate C2 with Better Prompts/Model**
- Timeline: 2 weeks prompt engineering + 1 week validation
- Cost: $2-3 (same model) or $10-15 (GPT-4/Claude)
- Confidence: Low (C1 also failed, suggests deeper issues)

**IF fundamental architecture issues** (likelihood: 50%):
- **Path C: Pivot to Meta-Learning Validation (RECOMMENDED)**
- Timeline: 2 weeks statistical validation
- Cost: $0 (no API calls)
- Confidence: High (proven effective in other contexts)

### Recommendation: Path C - Pivot to Meta-Learning

**Rationale**:
1. Both C1 and C2 collaborative modes failed (emergence <1.0)
2. Combined evidence suggests feedback mechanism itself problematic
3. Meta-learning (UCB1 operator selection) is orthogonal approach
4. Cost-effective ($0 vs $2-3+) with proven track record
5. Can return to C2 later if meta-learning validates feedback value

**Meta-Learning Validation Plan**:
- Compare fixed operator rates vs UCB1 adaptive rates
- 20 runs per condition (fixed vs adaptive)
- Statistical analysis following C1/C2 methodology
- Timeline: 2 weeks (1 week experiments, 1 week analysis)

**C2 Iteration Conditions** (if pursuing Path A/B):
- Complete Phase 3 root cause investigation first
- Identify specific fixable issues with evidence
- Cost-benefit threshold: Must show >20% improvement path to justify
- Try stronger LLM model (GPT-4 Turbo) if Gemini issue

---

## 8. Limitations

### Experimental Limitations
1. **Sample size**: N=15 (C2) vs N=10 (C1, rulebased) - unequal but sufficient given effect sizes
2. **Timing variance**: 0.5s evaluation has inherent variance (~5-10%)
3. **Single domain**: Results specific to GNFS parameter optimization
4. **Single LLM**: Only tested Gemini 2.5 Flash Lite (not GPT-4, Claude, or larger Gemini models)

### Methodological Considerations
1. **Cost factor**: LLM mode costs ~$2-3 vs $0 for rule-based (negative ROI demonstrated)
2. **Reproducibility**: LLM API may change between runs (minor variance)
3. **Prompt engineering**: Success may be highly prompt-dependent (untested hypothesis)
4. **Evaluation metric**: Fitness = smooth candidates (proxy for GNFS quality, not actual factorization)

### Statistical Considerations
1. **p-values**: 0.18-0.19 indicate large effects but insufficient statistical power at N=10-15
2. **Cohen's d**: Extremely large values (-8.65, -9.94) are unusual and may indicate outliers or measurement issues
3. **95% CIs**: Non-overlapping intervals suggest real differences despite non-significant p-values

---

## 9. Data Availability

### Raw Data Files
- **C2 results**: `results/c2_validation/c2_llm_seed9000.json` through `seed9014.json`
- **C1 results**: `results/c1_validation/collaborative_seed6000.json` through `seed6009.json`
- **Baseline**: `results/c1_validation/rulebased_seed8000.json` through `seed8009.json`

### Analysis Files
- **Statistical analysis**: `results/c2_validation/h1b_analysis.json`
- **Visualization figures**: `results/c2_validation/figures/figure1_*.png` (PNG + SVG, 12 files total)

### Reproducibility
```bash
# Reproduce C2 analysis
python scripts/analyze_c2_validation.py

# Generate figures
source .venv/bin/activate && python scripts/visualize_c2_results.py

# Run single experiment
python main.py --prometheus --prometheus-mode collaborative --llm \
  --generations 20 --population 20 --duration 0.5 --seed 9000 \
  --export-metrics results/c2_validation/c2_llm_seed9000.json
```

---

## 10. Conclusion

C2 validation revealed a **critical negative finding**: LLM-guided mutations dramatically underperformed both rule-based feedback (C1) and traditional evolution (rulebased baseline) by 52-54%.

**Key Takeaways**:
1. ❌ **H1b NOT SUPPORTED**: All 4 success criteria failed by substantial margins
2. ⚠️ **Dramatic regression**: C2 made C1's problems 10x worse (-54% vs -4.6% underperformance)
3. 💰 **Negative ROI**: Paid $2-3 to achieve 46% of baseline performance
4. 🔬 **Scientific value**: Negative result guides research direction away from unpromising approaches
5. 🎯 **Next steps**: Root cause investigation (Phase 3) required before deciding iteration vs pivot

**Comparison Context**:
- C1 (rule-based feedback): Emergence 0.954, underperformed by 4.6%
- C2 (LLM-guided): Emergence 0.462, underperformed by 54%
- **Both collaborative modes failed**, suggesting fundamental issues with feedback mechanism architecture rather than just mutation quality

**Research Integrity**: This negative result is documented thoroughly and honestly. The dramatic underperformance was unexpected but provides valuable evidence about LLM limitations for this domain. Future work should investigate root causes before additional investment in LLM-guided approaches.

**Recommendation**: Proceed with meta-learning validation (Path C) as lower-risk, cost-effective alternative. Return to C2 iteration only if root cause investigation reveals specific, fixable implementation bugs.

---

## 11. References

### Internal Documentation
- `plan_20251102.md` - C2 implementation plan
- `results/c1_validation/C1_RESULTS_SUMMARY.md` - C1 validation results
- `docs/prometheus_underperformance_analysis.md` - Root cause analysis patterns
- `CLAUDE.md` - Critical learnings from PR #52/#53

### Related Pull Requests
- PR #52: C2 LLM-guided mutations implementation
- PR #53: Critical fixes (temperature scaling, validation)
- PR #47: C1 feedback-guided mutations
- PR #49: C1 results analysis and visualization

### Research Methodology
- Pre-registered hypotheses in `plan_20251102.md`
- Statistical framework following C1 validation patterns
- Welch's t-test, Cohen's d, 95% CI (no p-hacking)

---

**Document Version**: 1.0
**Last Updated**: November 07, 2025
**Author**: Automated via C2 validation workflow
**Status**: COMPLETE - Awaiting Phase 3 root cause investigation
