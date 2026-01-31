#!/bin/bash
# Quick test run for Pokemon experiments
# This script runs a small subset of experiments for testing purposes

set -e

# Change to the project root directory (parent of bash directory)
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "Running quick test with GPT-4o-mini on brand dataset (subset of 3)..."

# Test with GPT-4o-mini on brand dataset (small subset)
python probing_pokemon.py \
    --mode openai \
    --model_name gpt-4o-mini \
    --num_runs 1 \
    --temperature 0.7 \
    --input_file ./experiments/brand/pokemon.csv \
    --output_dir ./results/test_brand_gpt4o \
    --subset_test true \
    --subset_size 3 \
    --max_workers 1

echo "Test completed. Results saved in ./results/test_brand_gpt4o/"

# Optional: Run evaluation on test results
read -p "Run evaluation on test results? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running evaluation..."
    python evaluation/evaluation_pokemon.py \
        --results-dir ./results \
        --out-dir ./evaluation/table_test \
        --seed 42

    echo "Evaluation completed. Tables saved in ./evaluation/table_test/"
fi