#!/bin/bash
# Run 5 C2 pilot experiments with generation history tracking
# Seeds: 9020-9024 (continuing from previous pilots 9015-9019)

set -e  # Exit on error

echo "🚀 Starting C2 Generation History Analysis - 5 pilots"
echo "Using seeds: 9020-9024"
echo "Target: 961730063 (same as C1 validation)"
echo ""

source .venv/bin/activate

for seed in 9020 9021 9022 9023 9024; do
    echo "=== [$(date +%H:%M:%S)] Running pilot with seed $seed ==="

    python main.py \
        --prometheus \
        --prometheus-mode collaborative \
        --llm \
        --generations 20 \
        --population 20 \
        --duration 0.5 \
        --seed $seed \
        --export-metrics results/c2_validation/c2_gen_analysis_seed${seed}.json

    echo "✅ Pilot $seed complete"
    echo ""
    sleep 2
done

echo "🎉 All 5 pilots complete!"
echo "Results saved to: results/c2_validation/c2_gen_analysis_seed*.json"
echo ""
echo "Run analysis with:"
echo "  python scripts/analyze_gen0_vs_gen1.py results/c2_validation/c2_gen_analysis_seed*.json"
