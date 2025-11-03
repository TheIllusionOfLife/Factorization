#!/bin/bash
# C2 Validation Experiment Runner
# Seeds 9000-9014 (15 runs)

source .venv/bin/activate

for seed in 9000 9001 9002 9003 9004 9005 9006 9007 9008 9009 9010 9011 9012 9013 9014; do
  echo "=== [$(date +%H:%M:%S)] Running C2 validation with seed $seed ==="

  python main.py --prometheus --prometheus-mode collaborative --llm \
    --generations 20 --population 20 --duration 0.5 --seed $seed \
    --export-metrics results/c2_validation/c2_llm_seed${seed}.json

  if [ $? -eq 0 ]; then
    echo "✅ Seed $seed completed successfully"
  else
    echo "❌ Seed $seed failed"
  fi

  # Brief pause between runs
  sleep 2
done

echo ""
echo "=== C2 Validation Complete ==="
echo "Results saved to: results/c2_validation/"
ls -lh results/c2_validation/c2_llm_seed*.json
