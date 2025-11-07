#!/bin/bash
# C2 Elite-Only LLM Pilot Experiment Runner
# Seeds 9015-9019 (5 runs) to validate Elite-Only LLM implementation

source .venv/bin/activate

echo "=== C2 Elite-Only LLM Pilot Experiments ==="
echo "Testing implementation with 5 experiments to verify:"
echo "  1. API call count ~80 per experiment (not 400)"
echo "  2. No 'API call limit reached' errors"
echo "  3. Fitness values reasonable (not degraded)"
echo ""

for seed in 9015 9016 9017 9018 9019; do
  echo "=== [$(date +%H:%M:%S)] Running pilot with seed $seed ==="

  python main.py --prometheus --prometheus-mode collaborative --llm \
    --generations 20 --population 20 --duration 0.5 --seed $seed \
    --export-metrics results/c2_validation/c2_elite_pilot_seed${seed}.json

  if [ $? -eq 0 ]; then
    echo "✅ Seed $seed completed successfully"

    # Quick check: count LLM errors in output
    if [ -f "results/c2_validation/c2_elite_pilot_seed${seed}.json" ]; then
      error_count=$(grep -o '"error"' results/c2_validation/c2_elite_pilot_seed${seed}.json | wc -l || echo 0)
      echo "   Error count in JSON: $error_count"
    fi
  else
    echo "❌ Seed $seed failed"
  fi

  # Brief pause between runs
  sleep 2
done

echo ""
echo "=== C2 Pilot Complete ==="
echo "Pilot results saved to: results/c2_validation/c2_elite_pilot_seed*.json"
ls -lh results/c2_validation/c2_elite_pilot_seed*.json 2>/dev/null || echo "No pilot files found"
