#!/bin/bash
# Re-run 5 C1 collaborative experiments with generation history tracking
# Seeds: 6000-6004 (matching original C1 validation seeds)

set -e  # Exit on error

echo "🚀 Re-running C1 Collaborative with Generation History - 5 pilots"
echo "Using seeds: 6000-6004 (matching original C1 validation)"
echo "Target: 961730063 (same as C1/C2 validation)"
echo ""

source .venv/bin/activate

for seed in 6000 6001 6002 6003 6004; do
    echo "=== [$(date +%H:%M:%S)] Running C1 collaborative with seed $seed ==="

    python main.py \
        --prometheus \
        --prometheus-mode collaborative \
        --generations 20 \
        --population 20 \
        --duration 0.5 \
        --seed $seed \
        --export-metrics results/c1_gen_history/collaborative_seed${seed}.json

    echo "✅ C1 pilot $seed complete"
    echo ""
    sleep 2
done

echo "🎉 All 5 C1 pilots complete!"
echo "Results saved to: results/c1_gen_history/collaborative_seed*.json"
echo ""
echo "Compare with C2:"
echo "  python scripts/analyze_fitness_growth.py results/c2_validation/c2_gen_analysis_seed*.json \\"
echo "    --max-gen 6 --compare results/c1_gen_history/collaborative_seed*.json \\"
echo "    --mode1-name 'C2 LLM' --mode2-name 'C1 Rule-based'"
