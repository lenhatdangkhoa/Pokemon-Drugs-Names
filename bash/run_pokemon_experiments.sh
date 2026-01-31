#!/bin/bash
# Pokemon Adversarial Hallucination Experiment Runner
# This script runs the complete Pokemon-Drugs-Names system step by step
#
# Usage: ./run_pokemon_experiments.sh [MODE] [OPTIONS]
#
# MODES:
#   openai     - Run experiments with OpenAI models (gpt-4o-mini, gpt-5-chat)
#   azure      - Run experiments with Azure OpenAI models
#   vllm       - Run experiments with local VLLM models
#   evaluation - Generate evaluation tables from existing results
#   all        - Run complete pipeline (experiments + evaluation)
#
# Examples:
#   ./run_pokemon_experiments.sh openai
#   ./run_pokemon_experiments.sh azure
#   ./run_pokemon_experiments.sh vllm
#   ./run_pokemon_experiments.sh evaluation
#   ./run_pokemon_experiments.sh all

set -e  # Exit on any error

# Configuration - assumes script is run from Pokemon-Drugs-Names root directory
POKEMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$POKEMON_DIR/results"
EVALUATION_DIR="$POKEMON_DIR/evaluation"
LOGS_DIR="$POKEMON_DIR/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Check if we're in the right directory
check_environment() {
    if [[ ! -f "probing_pokemon.py" ]]; then
        error "Not in Pokemon-Drugs-Names directory. Please run from the project root."
        exit 1
    fi

    # Create necessary directories
    mkdir -p "$RESULTS_DIR"
    mkdir -p "$EVALUATION_DIR/table"
    mkdir -p "$LOGS_DIR"

    log "Environment check passed"
}

# Run OpenAI experiments
run_openai_experiments() {
    log "Starting OpenAI experiments..."

    # GPT-4o-mini on brand dataset
    log "Running GPT-4o-mini on brand dataset..."
    python probing_pokemon.py \
        --mode openai \
        --model_name gpt-4o-mini \
        --num_runs 3 \
        --temperature 0.7 \
        --input_file ./experiments/brand/pokemon.csv \
        --output_dir ./results/brand_gpt4o \
        --max_workers 5 \
        2>&1 | tee "$LOGS_DIR/brand_gpt4o.log"

    # GPT-4o-mini on generic dataset
    log "Running GPT-4o-mini on generic dataset..."
    python probing_pokemon.py \
        --mode openai \
        --model_name gpt-4o-mini \
        --num_runs 3 \
        --temperature 0.7 \
        --input_file ./experiments/generic/pokemon.csv \
        --output_dir ./results/generic_gpt4o \
        --max_workers 5 \
        2>&1 | tee "$LOGS_DIR/generic_gpt4o.log"

    # GPT-5-chat on brand dataset (if available)
    if [[ "${RUN_GPT5:-false}" == "true" ]]; then
        log "Running GPT-5-chat on brand dataset..."
        python probing_pokemon.py \
            --mode openai \
            --model_name gpt-5-chat \
            --num_runs 3 \
            --temperature 0.7 \
            --input_file ./experiments/brand/pokemon.csv \
            --output_dir ./results/brand_gpt5 \
            --max_workers 3 \
            2>&1 | tee "$LOGS_DIR/brand_gpt5.log"

        # GPT-5-chat on generic dataset
        log "Running GPT-5-chat on generic dataset..."
        python probing_pokemon.py \
            --mode openai \
            --model_name gpt-5-chat \
            --num_runs 3 \
            --temperature 0.7 \
            --input_file ./experiments/generic/pokemon.csv \
            --output_dir ./results/generic_gpt5 \
            --max_workers 3 \
            2>&1 | tee "$LOGS_DIR/generic_gpt5.log"
    else
        warning "Skipping GPT-5-chat experiments (set RUN_GPT5=true to enable)"
    fi

    log "OpenAI experiments completed"
}

# Run Azure OpenAI experiments
run_azure_experiments() {
    log "Starting Azure OpenAI experiments..."

    # Check if Azure credentials are set
    if [[ -z "${AZURE_OPENAI_ENDPOINT:-}" ]]; then
        error "AZURE_OPENAI_ENDPOINT environment variable not set"
        error "Please set your Azure OpenAI endpoint:"
        error "export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'"
        exit 1
    fi

    # Azure GPT-4o-mini on brand dataset
    log "Running Azure GPT-4o-mini on brand dataset..."
    python probing_pokemon.py \
        --mode azure \
        --model_name azure-gpt-4o-mini \
        --num_runs 3 \
        --temperature 0.7 \
        --input_file ./experiments/brand/pokemon.csv \
        --output_dir ./results/brand_azure_gpt4o \
        --max_workers 5 \
        2>&1 | tee "$LOGS_DIR/brand_azure_gpt4o.log"

    # Azure GPT-4o-mini on generic dataset
    log "Running Azure GPT-4o-mini on generic dataset..."
    python probing_pokemon.py \
        --mode azure \
        --model_name azure-gpt-4o-mini \
        --num_runs 3 \
        --temperature 0.7 \
        --input_file ./experiments/generic/pokemon.csv \
        --output_dir ./results/generic_azure_gpt4o \
        --max_workers 5 \
        2>&1 | tee "$LOGS_DIR/generic_azure_gpt4o.log"

    # Azure GPT-5-chat (if available)
    if [[ "${RUN_AZURE_GPT5:-false}" == "true" ]]; then
        log "Running Azure GPT-5-chat on brand dataset..."
        python probing_pokemon.py \
            --mode azure \
            --model_name azure-gpt-5-chat \
            --num_runs 3 \
            --temperature 0.7 \
            --input_file ./experiments/brand/pokemon.csv \
            --output_dir ./results/brand_azure_gpt5 \
            --max_workers 3 \
            2>&1 | tee "$LOGS_DIR/brand_azure_gpt5.log"

        log "Running Azure GPT-5-chat on generic dataset..."
        python probing_pokemon.py \
            --mode azure \
            --model_name azure-gpt-5-chat \
            --num_runs 3 \
            --temperature 0.7 \
            --input_file ./experiments/generic/pokemon.csv \
            --output_dir ./results/generic_azure_gpt5 \
            --max_workers 3 \
            2>&1 | tee "$LOGS_DIR/generic_azure_gpt5.log"
    else
        warning "Skipping Azure GPT-5-chat experiments (set RUN_AZURE_GPT5=true to enable)"
    fi

    log "Azure experiments completed"
}

# Run VLLM experiments
run_vllm_experiments() {
    log "Starting VLLM experiments..."

    # Check if CUDA is available
    if ! command -v nvidia-smi &> /dev/null; then
        warning "nvidia-smi not found. Make sure CUDA is available for VLLM."
    fi

    # Gemma-3-27B on brand dataset
    if [[ "${RUN_GEMMA:-true}" == "true" ]]; then
        log "Running Gemma-3-27B on brand dataset..."
        CUDA_VISIBLE_DEVICES=0,1 python probing_pokemon.py \
            --mode vllm \
            --model_name google/gemma-3-27b-it \
            --number_gpus 2 \
            --batch_size 50 \
            --temperature 0.7 \
            --input_file ./experiments/brand/pokemon.csv \
            --output_dir ./results/brand_gemma \
            2>&1 | tee "$LOGS_DIR/brand_gemma.log"

        log "Running Gemma-3-27B on generic dataset..."
        CUDA_VISIBLE_DEVICES=0,1 python probing_pokemon.py \
            --mode vllm \
            --model_name google/gemma-3-27b-it \
            --number_gpus 2 \
            --batch_size 50 \
            --temperature 0.7 \
            --input_file ./experiments/generic/pokemon.csv \
            --output_dir ./results/generic_gemma \
            2>&1 | tee "$LOGS_DIR/generic_gemma.log"
    fi

    # Llama-3.3-70B on brand dataset
    if [[ "${RUN_LLAMA:-true}" == "true" ]]; then
        log "Running Llama-3.3-70B on brand dataset..."
        CUDA_VISIBLE_DEVICES=0,1 python probing_pokemon.py \
            --mode vllm \
            --model_name meta-llama/Llama-3.3-70B-Instruct \
            --number_gpus 2 \
            --batch_size 20 \
            --temperature 0.7 \
            --input_file ./experiments/brand/pokemon.csv \
            --output_dir ./results/brand_llama3 \
            2>&1 | tee "$LOGS_DIR/brand_llama3.log"

        log "Running Llama-3.3-70B on generic dataset..."
        CUDA_VISIBLE_DEVICES=0,1 python probing_pokemon.py \
            --mode vllm \
            --model_name meta-llama/Llama-3.3-70B-Instruct \
            --number_gpus 2 \
            --batch_size 20 \
            --temperature 0.7 \
            --input_file ./experiments/generic/pokemon.csv \
            --output_dir ./results/generic_llama3 \
            2>&1 | tee "$LOGS_DIR/generic_llama3.log"
    fi

    # Qwen3 on brand dataset
    if [[ "${RUN_QWEN:-true}" == "true" ]]; then
        log "Running Qwen3 on brand dataset..."
        CUDA_VISIBLE_DEVICES=0,1 python probing_pokemon.py \
            --mode vllm \
            --model_name Qwen/Qwen3-32B \
            --number_gpus 2 \
            --batch_size 50 \
            --temperature 0.7 \
            --input_file ./experiments/brand/pokemon.csv \
            --output_dir ./results/brand_qwen \
            2>&1 | tee "$LOGS_DIR/brand_qwen.log"

        log "Running Qwen3 on generic dataset..."
        CUDA_VISIBLE_DEVICES=0,1 python probing_pokemon.py \
            --mode vllm \
            --model_name Qwen/Qwen3-32B \
            --number_gpus 2 \
            --batch_size 50 \
            --temperature 0.7 \
            --input_file ./experiments/generic/pokemon.csv \
            --output_dir ./results/generic_qwen \
            2>&1 | tee "$LOGS_DIR/generic_qwen.log"
    fi

    log "VLLM experiments completed"
}

# Update suspicion labels (optional step)
update_suspicion_labels() {
    if [[ "${UPDATE_LABELS:-false}" == "true" ]]; then
        log "Updating suspicion labels..."

        # Find all result directories
        for result_dir in "$RESULTS_DIR"/*/; do
            if [[ -d "$result_dir" ]]; then
                dirname=$(basename "$result_dir")
                log "Updating labels in $dirname..."

                python src/update_suspicion_labels.py \
                    --results_dir "$result_dir" \
                    --dry_run false \
                    2>&1 | tee "$LOGS_DIR/update_labels_$dirname.log"
            fi
        done

        log "Suspicion labels updated"
    else
        info "Skipping suspicion label updates (set UPDATE_LABELS=true to enable)"
    fi
}

# Generate evaluation tables
run_evaluation() {
    log "Generating evaluation tables..."

    # Generate confabulation rate tables
    python evaluation/evaluation_pokemon.py \
        --results-dir ./results \
        --out-dir ./evaluation/table \
        --seed 42 \
        2>&1 | tee "$LOGS_DIR/evaluation.log"

    log "Evaluation tables generated in ./evaluation/table/"
    info "Generated files:"
    ls -la ./evaluation/table/
}

# Main execution logic
main() {
    local mode="${1:-help}"

    cd "$POKEMON_DIR"
    check_environment

    case "$mode" in
        "openai")
            info "Running OpenAI experiments only"
            run_openai_experiments
            update_suspicion_labels
            ;;

        "azure")
            info "Running Azure OpenAI experiments only"
            run_azure_experiments
            update_suspicion_labels
            ;;

        "vllm")
            info "Running VLLM experiments only"
            run_vllm_experiments
            update_suspicion_labels
            ;;

        "evaluation")
            info "Running evaluation only (requires existing results)"
            run_evaluation
            ;;

        "all")
            info "Running complete pipeline: OpenAI + Azure + VLLM experiments + evaluation"
            run_openai_experiments
            run_azure_experiments
            run_vllm_experiments
            update_suspicion_labels
            run_evaluation
            ;;

        "help"|"-h"|"--help")
            echo "Pokemon Adversarial Hallucination Experiment Runner"
            echo ""
            echo "Usage: $0 [MODE] [OPTIONS]"
            echo ""
            echo "MODES:"
            echo "  openai     - Run experiments with OpenAI models"
            echo "  azure      - Run experiments with Azure OpenAI models"
            echo "  vllm       - Run experiments with local VLLM models"
            echo "  evaluation - Generate evaluation tables from existing results"
            echo "  all        - Run complete pipeline (experiments + evaluation)"
            echo "  help       - Show this help message"
            echo ""
            echo "ENVIRONMENT VARIABLES:"
            echo "  RUN_GPT5=true          - Enable GPT-5-chat experiments"
            echo "  RUN_AZURE_GPT5=true    - Enable Azure GPT-5-chat experiments"
            echo "  RUN_GEMMA=true         - Enable Gemma experiments (default: true)"
            echo "  RUN_LLAMA=true         - Enable Llama experiments (default: true)"
            echo "  RUN_QWEN=true          - Enable Qwen experiments (default: true)"
            echo "  UPDATE_LABELS=true     - Enable suspicion label updates"
            echo "  AZURE_OPENAI_ENDPOINT  - Required for Azure experiments"
            echo ""
            echo "Examples:"
            echo "  $0 openai"
            echo "  $0 azure"
            echo "  $0 vllm"
            echo "  $0 evaluation"
            echo "  RUN_GPT5=true $0 all"
            echo ""
            exit 0
            ;;

        *)
            error "Unknown mode: $mode"
            echo "Run '$0 help' for usage information"
            exit 1
            ;;
    esac

    log "All tasks completed successfully!"
    info "Results saved in: $RESULTS_DIR"
    info "Evaluation tables saved in: $EVALUATION_DIR/table"
    info "Logs saved in: $LOGS_DIR"
}

# Run main function with all arguments
main "$@"