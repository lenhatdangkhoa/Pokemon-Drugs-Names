"""
Update JSON result files with new suspicion labels using HallucinationDetector.

This script processes all JSON files in the results directories and adds
suspicion_label_new by running hallucination detection on each response.

Usage:
    python update_suspicion_labels.py
"""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
# Try to find .env file in pokemon root (one level up from src/)
_pokemon_dir = Path(__file__).parent.parent
env_path = _pokemon_dir / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    # Fallback: try project root (two levels up)
    project_root = _pokemon_dir.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
    else:
        # Final fallback to find_dotenv (searches from current directory up)
        load_dotenv(find_dotenv(), override=True)

# Import hallucination detector - adjust path for src/ subdirectory
import sys
sys.path.insert(0, str(_pokemon_dir))
from src.hallucination_detector import HallucinationDetector
from src.online_rag import retrieve_drug_evidence

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Results directories to process (relative to pokemon/ directory)
RESULTS_DIRS = [
    str(_pokemon_dir / "results" / "brand_gemma"),
    str(_pokemon_dir / "results" / "brand_gpt4o"),
    str(_pokemon_dir / "results" / "brand_llama"),
    str(_pokemon_dir / "results" / "brand_qwen"),
    str(_pokemon_dir / "results" / "generic_gemma"),
    str(_pokemon_dir / "results" / "generic_gpt_4o"),
    str(_pokemon_dir / "results" / "generic_llama"),
    str(_pokemon_dir / "results" / "generic_qwen"),
]


def load_json_file(file_path: Path) -> Dict:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(file_path: Path, data: Dict):
    """Save JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_json_file(
    file_path: Path,
    detector: HallucinationDetector,
    *,
    force: bool = False,
    use_rag: bool = False,
    use_pokemon: bool = False,
    max_workers: int = 1,
) -> tuple[int, int]:
    """
    Process a single JSON file and add suspicion_label_new to all answers.
    
    Returns:
        Tuple of (total_processed, total_errors)
    """
    try:
        data = load_json_file(file_path)
        total_processed = 0
        total_errors = 0

        # Build judging tasks first so we can show meaningful progress.
        # We fetch RAG evidence once per case (if enabled), then judge each answer.
        tasks: List[Tuple[str, int, str, str, str]] = []
        # (case_id, answer_idx, response_text, pokemon_name, original_for_judge)
        per_case_answers: Dict[str, List[Dict[str, Any]]] = {}

        for case_id, case_data in data.items():
            pokemon_name = case_data.get("pokemon", "")
            original_drug_list = case_data.get("query -> drug list", "")
            original_for_judge = original_drug_list

            if use_rag or use_pokemon:
                try:
                    evidence = retrieve_drug_evidence(original_drug_list, use_pokemon=use_pokemon)
                except Exception as e:
                    logging.warning(f"RAG retrieval failed for {file_path}, case {case_id}: {e}")
                    evidence = ""
                if evidence:
                    original_for_judge = original_drug_list + "\n\n[Reference: " + evidence + "]"

            answers = case_data.get("answers", []) or []
            per_case_answers[str(case_id)] = answers

            for answer_idx, answer in enumerate(answers):
                response_text = answer.get("response", "")
                if not response_text:
                    logging.warning(f"Empty response in {file_path}, case {case_id}")
                    continue

                if "suspicion_label_new" in answer and not force:
                    continue

                tasks.append((str(case_id), answer_idx, response_text, pokemon_name, original_for_judge))

        if not tasks:
            # Nothing to do; still re-save to keep behavior predictable
            save_json_file(file_path, data)
            return 0, 0

        def _judge_one(task: Tuple[str, int, str, str, str]) -> Tuple[str, int, Optional[int], Optional[bool], Optional[str]]:
            case_id, answer_idx, response_text, pokemon_name, original_for_judge = task
            try:
                suspicion_label_new, suspicion_detected_new = detector.detect_hallucination(
                    response_text=response_text,
                    pokemon_name=pokemon_name,
                    original_drug_list=original_for_judge,
                )
                return case_id, answer_idx, suspicion_label_new, suspicion_detected_new, None
            except Exception as e:
                return case_id, answer_idx, 0, False, str(e)

        # Progress bar similar to probing_pokemon.py: show file + current case.
        pbar = tqdm(total=len(tasks), desc=f"Judging ({file_path.name})")
        workers = max(1, int(max_workers))

        if workers == 1:
            for t in tasks:
                case_id, answer_idx, label, detected, err = _judge_one(t)
                ans = per_case_answers[case_id][answer_idx]
                ans["suspicion_label_new"] = label
                ans["suspicion_detected_new"] = detected
                if err:
                    total_errors += 1
                    logging.error(f"Error processing answer in {file_path}, case {case_id}: {err}")
                else:
                    total_processed += 1
                pbar.update(1)
                pbar.set_postfix({"case": case_id})
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_task = {executor.submit(_judge_one, t): t for t in tasks}
                for future in as_completed(future_to_task):
                    case_id, answer_idx, label, detected, err = future.result()
                    ans = per_case_answers[case_id][answer_idx]
                    ans["suspicion_label_new"] = label
                    ans["suspicion_detected_new"] = detected
                    if err:
                        total_errors += 1
                        logging.error(f"Error processing answer in {file_path}, case {case_id}: {err}")
                    else:
                        total_processed += 1
                    pbar.update(1)
                    pbar.set_postfix({"case": case_id, "workers": workers})

        pbar.close()
        
        # Save updated JSON file
        save_json_file(file_path, data)
        return total_processed, total_errors
        
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return 0, 1


def main():
    parser = argparse.ArgumentParser(
        description="Update JSON result files with new suspicion labels"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to use for hallucination detection (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - don't save changes"
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=RESULTS_DIRS,
        help="List of results directories to process"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if suspicion_label_new already exists"
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="Augment judge context with online RAG evidence for each case",
    )
    parser.add_argument(
        "--use-pokemon",
        action="store_true",
        help="Append PokéAPI RAG (rag_cache_pokemon.json). Can be used alone or with --use-rag; if both, PokéAPI is used.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel judge workers (default: 1). Increase to speed up, but may hit rate limits.",
    )
    args = parser.parse_args()
    
    # Initialize detector
    logging.info(f"Initializing HallucinationDetector with model: {args.model}")
    detector = HallucinationDetector(model=args.model)
    
    # Collect all JSON files (excluding experiment_summary.json)
    json_files = []
    for results_dir in args.results_dirs:
        results_path = Path(results_dir)
        if not results_path.exists():
            logging.warning(f"Results directory not found: {results_dir}")
            continue
        
        for json_file in results_path.glob("*.json"):
            if json_file.name != "experiment_summary.json":
                json_files.append(json_file)
    
    logging.info(f"Found {len(json_files)} JSON files to process")
    
    # Process files
    total_processed = 0
    total_errors = 0
    
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        logging.info(f"Processing: {json_file}")
        
        if args.dry_run:
            logging.info(f"DRY RUN: Would process {json_file}")
            continue
        
        processed, errors = process_json_file(
            json_file,
            detector,
            force=args.force,
            use_rag=args.use_rag,
            use_pokemon=args.use_pokemon,
            max_workers=args.max_workers,
        )
        total_processed += processed
        total_errors += errors
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Processing complete!")
    logging.info(f"Total responses processed: {total_processed}")
    logging.info(f"Total errors: {total_errors}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
