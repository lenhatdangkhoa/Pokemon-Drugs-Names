"""
Pokemon Adversarial Hallucination Experiment

This script evaluates LLMs' susceptibility to adversarial hallucination attacks
by testing if they can distinguish between real medications and Pokemon names
embedded in clinical medication lists.

Based on methodology from:
Omar et al. Multi-model assurance analysis showing large language models are
highly vulnerable to adversarial hallucination attacks during clinical decision support.
Communications Medicine, 2025.

TEMPERATURE SETTINGS - CRITICAL DESIGN DECISIONS:
========================================================
1. GENERATION TASKS (Primary LLM): Use configurable temperature (default 0.7)
   - Allows creative variation in responses
   - Tests robustness across different outputs
   - Simulates real-world usage scenarios

2. JUDGE/EVALUATION TASKS (LLM-as-a-Judge): ALWAYS use temperature=0.0
   - Ensures deterministic, reproducible judgments
   - Eliminates randomness in evaluation results
   - Enables reliable comparison across experiments
   - Prevents judge from being "creative" in classification

3. ADVANCED JUDGE PROMPTING: Chain-of-thought evaluation
   - Step-by-step analysis of response content
   - Structured rubric with specific examples
   - Explicit reasoning before final classification
   - Based on Verdict library best practices for LLM-as-a-judge

WHY T=0 FOR JUDGING:
- Consistency: Same input → same judgment every time
- Reliability: Removes stochastic variation from metrics
- Reproducibility: Scientific rigor in evaluation frameworks
- Objectivity: Pure logical assessment, not creative interpretation

python probing_pokemon.py \
  --model_name azure-gpt-4o-mini \
  --num_runs 3 \
  --temperature 1.2 \
  --top_p 0.95 \
  --max_new_tokens 512 \
  --input_file ./data/brand/pokemon.csv \
  --output_dir ./results/brand_gpt4o \
  --subset_test True \
  --subset_size 3
  --request_delay 0.01

# Running GPT 5
python probing_pokemon.py \
  --mode azure \
  --model_name azure-gpt-5-chat \
  --num_runs 3 \
  --temperature 1.2 \
  --top_p 0.95 \
  --max_new_tokens 512 \
  --input_file ./data/brand/pokemon.csv \
  --output_dir ./results/brand_gpt5 \
  --max_workers 5

python probing_pokemon.py \
  --mode azure \
  --model_name azure-gpt-5-chat \
  --num_runs 3 \
  --temperature 1.2 \
  --top_p 0.95 \
  --max_new_tokens 512 \
  --input_file ./data/generic/pokemon.csv \
  --output_dir ./results/generic_gpt5 \
  --max_workers 5

  

python probing_pokemon.py \
  --model_name gpt-4o-mini \
  --num_runs 3

python probing_pokemon.py --model_name azure-gpt-4o-mini --num_runs 3 --subset_test True --subset_size 10

# Subset testing examples (useful for quick testing)
python probing_pokemon.py \
  --model_name azure-gpt-5-chat \
  --num_runs 1 \
  --subset_test True \
  --subset_size 3

python probing_pokemon.py \
  --mode azure \
  --model_name azure-gpt-5-chat \
  --num_runs 1 \
  --subset_test True \
  --subset_size 3 \
  --max_workers 1 \
  --request_delay 0.5

# VLLM Mode Examples
CUDA_VISIBLE_DEVICES=0,1 python probing_pokemon.py \
  --mode vllm \
  --model_name google/gemma-3-27b-it \
  --number_gpus 2 \
  --batch_size 50 \
  --temperature 0.7

CUDA_VISIBLE_DEVICES=2,3 python probing_pokemon.py \
  --mode vllm \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --number_gpus 2 \
  --batch_size 20 \
  --temperature 0.7

CUDA_VISIBLE_DEVICES=0,1 nohup python probing_pokemon.py \
  --mode vllm \
  --model_name Qwen/Qwen3-32B \
  --number_gpus 2 \
  --batch_size 50 \
  --temperature 0.7 \
  --input_file ./data/brand/pokemon.csv \
  --output_dir ./results/brand_qwen > ./outputs/brand_qwen.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 nohup python probing_pokemon.py \
--mode vllm \
--model_name Qwen/Qwen3-32B \
--number_gpus 2 \
--batch_size 50 \
--temperature 0.7 \
--input_file ./data/generic/pokemon.csv \
--output_dir ./results/generic_qwen > ./outputs/generic_qwen.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1 python probing_pokemon.py \
  --mode vllm \
  --model_name google/gemma-3-27b-it \
  --number_gpus 2 \
  --batch_size 50 \
  --temperature 0.7 \
  --input_file ./data/generic/pokemon.csv \
  --output_dir ./results/generic_gemma > ./outputs/generic_gemma.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 python probing_pokemon.py \
  --mode vllm \
  --model_name google/gemma-3-27b-it \
  --number_gpus 2 \
  --batch_size 50 \
  --temperature 0.7 \
  --input_file ./data/brand/pokemon.csv \
  --output_dir ./results/brand_gemma > ./outputs/brand_gemma.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1 nohup python probing_pokemon.py \
  --mode vllm \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --number_gpus 2 \
  --batch_size 20 \
  --temperature 0.7 \
  --input_file ./data/generic/pokemon.csv \
  --output_dir ./results/generic_llama3 > ./outputs/generic_llama3.log 2>&1 &


CUDA_VISIBLE_DEVICES=2,3 nohup python probing_pokemon.py \
--mode vllm \
--model_name meta-llama/Llama-3.3-70B-Instruct \
--number_gpus 2 \
--batch_size 20 \
--temperature 0.7 \
--input_file ./data/brand/pokemon.csv \
--output_dir ./results/brand_llama3 > ./outputs/brand_llama3.log 2>&1 &



nohup python update_suspicion_labels.py > ./outputs/update_label.log 2>&1 &
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import random
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

# Conditional imports
try:
    from transformers import HfArgumentParser
    HF_PARSER_AVAILABLE = True
except ImportError:
    HF_PARSER_AVAILABLE = False
    print("⚠️  transformers not available, using basic argument parsing")

# Local imports - adjust path for pokemon root directory
from pathlib import Path
# Add current directory to path to import from src/
sys.path.insert(0, str(Path(__file__).parent))

# Import from src package
from src.constants import CONDITIONS
from src.data_loader import load_pokemon_data
from src.experiment_runner import run_experiment_condition, run_vllm_processing_pokemon
from src.metrics import calculate_metrics
from src.results_formatter import display_results_table
from src.utils import clear_gpu_memory, print_gpu_allocation
from src.vllm_setup import setup_vllm_mode, validate_openai_args, validate_azure_args, validate_vllm_args

#####################################################################
#                       Environment Configuration                    #
#####################################################################
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv(find_dotenv())
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

#####################################################################
#                       Argument Definitions                        #
#####################################################################

@dataclass
class ScriptArguments:
    # Mode selection
    mode: Optional[str] = field(default="openai", metadata={"help": "Mode: 'openai' for OpenAI API, 'azure' for Azure OpenAI API, 'vllm' for local VLLM inference"})
    model_name: Optional[str] = field(default="gpt-4o-mini", metadata={"help": "Model name to test (e.g., 'gpt-4o-mini', 'gpt-5-chat', 'azure-gpt-4o-mini', 'azure-gpt-5-chat')"})

    # Generation parameters
    temperature: Optional[float] = field(default=0.7, metadata={"help": "Temperature for default runs"})
    top_p: Optional[float] = field(default=0.95, metadata={"help": "Top-p for sampling"})
    max_new_tokens: Optional[int] = field(default=512, metadata={"help": "Max tokens for responses"})

    # LoRA configuration (VLLM only)
    lora_path: Optional[str] = field(default=None, metadata={"help": "Path to the LoRA model (VLLM only)"})

    # VLLM-specific parameters
    presence_penalty: Optional[float] = field(default=0.0, metadata={"help": "presence penalty for sampling (VLLM only)"})
    frequency_penalty: Optional[float] = field(default=0.0, metadata={"help": "frequency penalty for sampling (VLLM only)"})
    repetition_penalty: Optional[float] = field(default=1.0, metadata={"help": "repetition penalty for sampling (VLLM only)"})

    # Data configuration
    input_file: Optional[str] = field(default="./data/brand/pokemon.csv", metadata={"help": "Path to Pokemon CSV (relative to pokemon/ directory)"})
    output_dir: Optional[str] = field(default="./results/brand_gemma", metadata={"help": "Output directory for condition-specific JSON files (relative to pokemon/ directory)"})

    # Experiment configuration
    num_runs: Optional[int] = field(default=3, metadata={"help": "Number of runs per condition"})
    subset_test: Optional[bool] = field(default=False, metadata={"help": "Use subset for testing (e.g., --subset_test True --subset_size 3 for quick testing with 3 data points)"})
    subset_size: Optional[int] = field(default=10, metadata={"help": "Subset size if subset_test=True (e.g., 3 for quick testing)"})

    # Hallucination detection
    judge_model: Optional[str] = field(default=None, metadata={"help": "Model for LLM-as-a-Judge hallucination detection (defaults to model_name if None)"})

    # Logging
    log_with: Optional[str] = field(default="none", metadata={"help": "Logging tool (none, wandb)"})
    seed: Optional[int] = field(default=42, metadata={"help": "Random seed"})

    # Processing configuration
    batch_size: Optional[int] = field(default=10, metadata={"help": "Batch size for processing (VLLM only)"})
    number_gpus: Optional[int] = field(default=2, metadata={"help": "Number of GPUs to use (VLLM only)"})
    max_workers: Optional[int] = field(default=10, metadata={"help": "Max parallel workers for OpenAI/Azure mode (default: 1 for Azure, 10 for OpenAI to avoid rate limits)"})
    request_delay: Optional[float] = field(default=0.0, metadata={"help": "Delay in seconds between API requests (useful for rate limiting, default: 0.0)"})

    # Bootstrap configuration
    bootstrap_size: Optional[int] = field(default=1000, metadata={"help": "Number of bootstrap samples for CI calculation"})


def main():
    """Main experiment function."""
    # Parse arguments
    if HF_PARSER_AVAILABLE and len(sys.argv) > 1:
        parser = HfArgumentParser(ScriptArguments)
        args = parser.parse_args_into_dataclasses()[0]
    else:
        # Use defaults if no arguments provided or parser not available
        args = ScriptArguments()
    
    # Auto-detect Azure mode if model name starts with "azure-" (similar to benchmark script)
    if args.model_name.startswith("azure-") and args.mode == "openai":
        args.mode = "azure"
        logging.info(f"Auto-detected Azure mode based on model name: {args.model_name}")
    
    # Display configuration
    print("\n" + "="*60)
    print("POKEMON ADVERSARIAL HALLUCINATION EXPERIMENT")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_name}")
    if args.mode == "azure":
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-azure-openai-endpoint.openai.azure.com/")
        print(f"Azure Endpoint: {endpoint}")
    elif args.mode == "vllm":
        print(f"Number of GPUs: {args.number_gpus}")
        print(f"Batch size: {args.batch_size}")
        if args.lora_path:
            print(f"LoRA path: {args.lora_path}")
    judge_model_display = args.judge_model if args.judge_model else args.model_name
    print(f"Hallucination Detection: LLM-as-a-Judge (using {judge_model_display})")
    print(f"Temperature (default): {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Runs per condition: {args.num_runs}")
    if args.subset_test:
        print(f"Subset test: ENABLED (using {args.subset_size} data points)")
    else:
        print(f"Subset test: DISABLED (using all data)")
    if args.mode == "openai" or args.mode == "azure":
        max_workers = 1 if args.mode == "azure" else 10 if args.max_workers is None else args.max_workers
        print(f"Max workers: {max_workers} ({'sequential' if max_workers == 1 else 'parallel'})")
        if args.request_delay > 0:
            print(f"Request delay: {args.request_delay}s between requests")
    print("="*60 + "\n")
    
    # Validate configuration based on mode
    if args.mode == "openai":
        validate_openai_args(args)
    elif args.mode == "azure":
        validate_azure_args(args)
    elif args.mode == "vllm":
        validate_vllm_args(args)
        # Print GPU allocation info
        print_gpu_allocation()
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'openai', 'azure', or 'vllm'")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Reset cost tracker at start of experiment
    from src.client_factory import reset_cost_tracker, get_cost_summary
    reset_cost_tracker()
    
    # Resolve output directory path relative to pokemon directory if it's a relative path
    if not os.path.isabs(args.output_dir):
        # Get the pokemon directory (where this script is located)
        script_dir = Path(__file__).parent
        resolved_output_dir = script_dir / args.output_dir
        # Normalize the path (resolve .. and .)
        args.output_dir = str(resolved_output_dir.resolve())
    
    # Load data
    pokemon_cases = load_pokemon_data(args.input_file, args.subset_test, args.subset_size)

    # Initialize results dictionary to group by condition and case
    results_by_condition = {condition: {} for condition in CONDITIONS}

    # Run experiment based on mode
    if args.mode == "openai" or args.mode == "azure":
        # OpenAI/Azure mode - parallel processing with ThreadPoolExecutor
        mode_display = "Azure OpenAI" if args.mode == "azure" else "OpenAI"
        total_experiments = len(pokemon_cases) * len(CONDITIONS) * args.num_runs
        
        # Determine max_workers: default to 1 for Azure (sequential) to avoid rate limits, 10 for OpenAI
        if args.max_workers is None:
            max_workers = 1 if args.mode == "azure" else 10
        else:
            max_workers = args.max_workers
        
        processing_mode = "sequential" if max_workers == 1 else f"parallel ({max_workers} workers)"
        pbar = tqdm(total=total_experiments, desc=f"Running {mode_display} experiments ({processing_mode})")

        # Create list of all tasks to run
        tasks = []
        for pokemon_case in pokemon_cases:
            for condition in CONDITIONS:
                for run_num in range(1, args.num_runs + 1):
                    tasks.append((pokemon_case, condition, run_num))

        # Execute tasks using ThreadPoolExecutor (sequential if max_workers=1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(run_experiment_condition, pokemon_case, condition, run_num, args): (pokemon_case, condition, run_num)
                for pokemon_case, condition, run_num in tasks
            }

            # Process completed tasks
            import time
            for future in as_completed(future_to_task):
                pokemon_case, condition, run_num = future_to_task[future]
                try:
                    result = future.result()

                    # Group results by condition and case_id
                    case_id = pokemon_case["case_id"]
                    if case_id not in results_by_condition[condition]:
                        results_by_condition[condition][case_id] = {
                            "query -> drug list": pokemon_case["pokemon_list"],
                            "pokemon": pokemon_case["pokemon_name"],
                            "answers": []
                        }

                    results_by_condition[condition][case_id]["answers"].append({
                        "response": result["llm_response"],
                        "suspicion_detected": result["suspicion_detected"],
                        "suspicion_label": result["suspicion_label"]
                    })

                    pbar.update(1)
                    pbar.set_postfix({
                        "case": pokemon_case["case_id"],
                        "condition": condition,
                        "run": run_num
                    })
                    
                    # Add delay between requests if specified (helps with rate limiting)
                    if args.request_delay > 0:
                        time.sleep(args.request_delay)

                except Exception as e:
                    logging.error(f"Task failed for case {pokemon_case['case_id']}, condition {condition}, run {run_num}: {e}")
                    pbar.update(1)
                    # Still add delay even on error to avoid hammering the API
                    if args.request_delay > 0:
                        time.sleep(args.request_delay)

        pbar.close()

    elif args.mode == "vllm":
        # VLLM mode - batch processing
        print("Setting up VLLM model...")
        model, tokenizer, sampling_params, judge_sampling_params, lora_request = setup_vllm_mode(args)

        try:
            results_by_condition = run_vllm_processing_pokemon(
                args, model, tokenizer, sampling_params, judge_sampling_params, lora_request
            )
        finally:
            # Clean up GPU memory
            try:
                del model
                clear_gpu_memory()
            except NameError:
                pass

    # Save results to condition-specific JSON files
    for condition, cases in results_by_condition.items():
        output_file = os.path.join(args.output_dir, f"{condition}.json")
        os.makedirs(args.output_dir, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)

        print(f"Saved {condition} results to: {output_file}")

    # Convert to flat list for metrics calculation
    all_results = []
    for condition, cases in results_by_condition.items():
        for case_id, case_data in cases.items():
            for run_num, answer_data in enumerate(case_data["answers"], 1):
                all_results.append({
                    "case_id": case_id,
                    "pokemon_name": case_data["pokemon"],
                    "condition": condition,
                    "run_number": run_num,
                    "llm_response": answer_data["response"],
                    "suspicion_detected": answer_data["suspicion_detected"],
                    "suspicion_label": answer_data["suspicion_label"]
                })

    # Calculate and display metrics with bootstrap CI
    metrics = calculate_metrics(all_results, bootstrap_size=args.bootstrap_size)
    
    # Display results table
    display_results_table(metrics, args.model_name)
    
    print(f"Results saved to: {args.output_dir}/")
    print(f"Files: {', '.join([f'{c}.json' for c in CONDITIONS])}")
    print("="*90 + "\n")

    # Save summary metrics
    summary_file = os.path.join(args.output_dir, "experiment_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(summary_file, "w") as f:
        judge_model_display = args.judge_model if args.judge_model else args.model_name
        json.dump({
            "model_tested": args.model_name,
            "hallucination_detection": "llm_as_a_judge",
            "judge_model": judge_model_display,
            "total_cases": len(pokemon_cases),
            "bootstrap_size": args.bootstrap_size,
            "confidence_interval": 0.95,
            "metrics": metrics
        }, f, indent=2)

    print(f"Summary saved to: {summary_file}\n")
    
    # Display cost summary for Azure/OpenAI modes
    if args.mode == "openai" or args.mode == "azure":
        cost_summary = get_cost_summary()
        print("\n" + "="*60)
        print("TOKEN USAGE & COST SUMMARY")
        print("="*60)
        print(f"Model: {cost_summary['model']}")
        print(f"Total API Requests: {cost_summary['total_requests']}")
        print(f"Input Tokens: {cost_summary['total_input_tokens']:,}")
        print(f"Output Tokens: {cost_summary['total_output_tokens']:,}")
        print(f"Total Tokens: {cost_summary['total_tokens']:,}")
        print(f"Estimated Cost: ${cost_summary['estimated_cost_usd']:.4f} USD")
        print("="*60 + "\n")
        
        # Add cost info to summary file
        with open(summary_file, "r") as f:
            summary_data = json.load(f)
        summary_data["token_usage"] = cost_summary
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)


if __name__ == "__main__":
    main()
