#!/bin/bash
# Run zero-shot and few-shot baselines on Spider dev set
set -euo pipefail

MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR="results"

mkdir -p "$OUTPUT_DIR"

echo "=== Running Zero-Shot Baseline ==="
python -m src.eval.run_eval \
    --model "$MODEL" \
    --mode zero-shot \
    --split dev \
    --output "$OUTPUT_DIR"

echo ""
echo "=== Running 5-Shot Baseline ==="
python -m src.eval.run_eval \
    --model "$MODEL" \
    --mode few-shot \
    -k 5 \
    --split dev \
    --output "$OUTPUT_DIR"

echo ""
echo "=== Running Error Analysis on Zero-Shot ==="
python -m src.eval.error_analysis \
    --predictions "$OUTPUT_DIR/eval_dev_zero-shot.json"

echo ""
echo "=== Baselines Complete ==="
echo "Results saved to $OUTPUT_DIR/"
