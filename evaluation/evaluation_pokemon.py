"""
Generate confabulation-rate tables from Pokemon experiment outputs.

Reads JSON result files directly and calculates metrics based on suspicion_label.
Uses overall hallucinations: suspicion_label ∈ {0,1} → hallucination, suspicion_label = 2 → no hallucination.

Generates a single combined table with three confabulation types:
1. Overall confabulations (suspicion_label = 0 and 1)
2. Inherited confabulations (suspicion_label = 0)
3. Epistemic confabulations (suspicion_label = 1)

Reads JSON files from:
  - ./results

Expected file names:
  - drug_dosing.json, drug_dosing_mitigation.json, drug_dosing_temp0.json
  - medication_indication.json, medication_indication_mitigation.json, medication_indication_temp0.json

and writes tables to:
  - ./evaluation/table

Outputs:
  - confabulation_rates_table.md/csv/tex
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# Use relative paths from pokemon root directory
DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results"
DEFAULT_OUT_DIR = Path(__file__).parent.parent / "evaluation" / "table"


# Prompt -> condition mapping
# Each condition can have multiple alternative file names to handle naming inconsistencies
PROMPTS: List[Tuple[str, str, List[Tuple[str, List[str]]]]] = [
    (
        "Drug Dosing Prompt",
        "dosing",
        [
            ("Default", ["drug_dosing", "default"]),
            ("Default + Mitigation", ["drug_dosing_mitigation", "mitigation"]),
            ("Temp 0", ["drug_dosing_temp0", "temp0"]),
        ],
    ),
    (
        "Medication Indication Prompt",
        "medication_indication",
        [
            ("Default", ["medication_indication"]),
            ("Default + Mitigation", ["medication_indication_mitigation"]),
            ("Temp 0", ["medication_indication_temp0"]),
        ],
    ),
]

DATASETS: List[Tuple[str, str]] = [
    ("generic drug dataset", "generic"),
    ("brand drug dataset", "brand"),
]

# Desired column order for the final table.
MODEL_COL_ORDER = [
    "gpt-5-chat",
    "GPT-4o-mini",
    "Llama-3.3-70B-Instruct",
    "Gemma-3-27B-IT",
    "Qwen3"
]


def _safe_read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON at {path}, got {type(data)}")
    return data


def _normalize_model_name(model_tested: str) -> str:
    """
    Normalize model name from directory name or experiment summary.
    
    Handles directory names like "brand_gemma", "generic_llama3", etc.
    Also handles full model names from experiment summaries.
    
    Args:
        model_tested: Model name from directory or experiment summary
        
    Returns:
        Normalized model display name
    """
    s = (model_tested or "").strip()
    s_low = s.lower()
    
    # Remove dataset prefix if present (e.g., "brand_gemma" -> "gemma")
    if s_low.startswith("brand_") or s_low.startswith("generic_"):
        s_low = s_low.split("_", 1)[1] if "_" in s_low else s_low

    # GPT-4o-mini variants (OpenAI / Azure)
    if "gpt-4o-mini" in s_low or "gpt4o" in s_low or "gpt-4o" in s_low:
        return "GPT-4o-mini"

    # GPT-5-chat variants (OpenAI / Azure)
    if "gpt-5-chat" in s_low or "gpt5" in s_low or "gpt-5" in s_low:
        return "gpt-5-chat"

    # Llama (handles "llama3", "llama-3", "llama-3.3", etc.)
    if "llama-3.3-70b-instruct" in s_low or "llama-3.3" in s_low or "llama3" in s_low or "llama" in s_low:
        return "Llama-3.3-70B-Instruct"

    # Gemma (handles "gemma", "gemma-3", etc.)
    if "gemma-3-27b-it" in s_low or "gemma-3" in s_low or "gemma" in s_low:
        return "Gemma-3-27B-IT"

    # Qwen (handles "qwen", "qwen3", etc.)
    if "qwen3" in s_low or "qwen" in s_low:
        return "Qwen3"

    return s or "Unknown"


def _format_rate_ci(rate: float, ci_lower: float, ci_upper: float) -> str:
    """
    Match paper-style formatting, e.g.:
      96.4% [94.4, 98.1]
    """
    r = rate * 100.0
    lo = ci_lower * 100.0
    hi = ci_upper * 100.0
    return f"{r:.1f}% [{lo:.1f}, {hi:.1f}]"


def _discover_model_dirs(results_dir: Path) -> List[Path]:
    """
    Discover model directories that match the pattern brand_* or generic_*.
    
    Args:
        results_dir: Root directory containing model subdirectories
        
    Returns:
        Sorted list of Path objects for model directories
    """
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    model_dirs = []
    for p in results_dir.iterdir():
        if not p.is_dir():
            continue
        dirname_lower = p.name.lower()
        # Only include directories that match brand_* or generic_* pattern
        if dirname_lower.startswith("brand_") or dirname_lower.startswith("generic_"):
            model_dirs.append(p)
    
    return sorted(model_dirs)


def _dataset_from_dirname(dirname: str) -> Optional[str]:
    d = dirname.lower()
    if d.startswith("generic_"):
        return "generic"
    if d.startswith("brand_"):
        return "brand"
    return None


def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for hallucination rate.
    
    Args:
        data: List of case-level average hallucination rates (values between 0 and 1)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound) for the hallucination rate CI
    """
    if not data:
        return (0.0, 0.0)
    
    n = len(data)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # Sample cases with replacement
        bootstrap_sample = random.choices(data, k=n)
        # Calculate mean of case-level averages
        bootstrap_mean = sum(bootstrap_sample) / n if n > 0 else 0.0
        bootstrap_means.append(bootstrap_mean)
    
    # Calculate percentiles for CI
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return (lower_bound, upper_bound)


def _find_json_file(model_dir: Path, condition_alternatives: List[str]) -> Optional[Path]:
    """
    Find the first existing JSON file from a list of condition alternatives.
    
    Args:
        model_dir: Directory containing JSON files
        condition_alternatives: List of possible condition names to try
        
    Returns:
        Path to the first existing JSON file, or None if none found
    """
    for condition in condition_alternatives:
        json_file = model_dir / f"{condition}.json"
        if json_file.exists():
            return json_file
    return None


def _get_suspicion_label(answer: Dict[str, Any]) -> Optional[int]:
    """
    Get the suspicion label from an answer.
    Uses overall hallucinations: suspicion_label ∈ {0,1} → hallucination, suspicion_label = 2 → no hallucination.
    
    Args:
        answer: Answer dictionary from JSON data
        
    Returns:
        The suspicion label (0, 1, or 2) or None if not found
    """
    # Use suspicion_label only (overall hallucinations)
    return answer.get("suspicion_label")


def _calculate_metrics_from_json(
    model_dir: Path,
    condition_alternatives: List[str],
    label_filter: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculate metrics from JSON files for a specific condition.
    
    Args:
        model_dir: Directory containing JSON files
        condition_alternatives: List of possible condition names (e.g., ["drug_dosing", "default"])
        label_filter: If provided, only count answers with this suspicion_label
                     None = Overall (count 0 and 1 as confabulations)
                     0 = Inherited confabulations only
                     1 = Epistemic confabulations only
        
    Returns:
        Dictionary with metrics including hallucination_rate, ci_lower, ci_upper
    """
    json_file = _find_json_file(model_dir, condition_alternatives)
    if json_file is None:
        return None
    
    data = _safe_read_json(json_file)
    
    # Group results by case_id to calculate case-level averages
    cases_dict = {}
    for case_id, case_data in data.items():
        case_results = []
        for answer in case_data.get("answers", []):
            suspicion_label = _get_suspicion_label(answer)
            
            if suspicion_label is None:
                continue
            
            # For Overall (label_filter=None): count where label is 0 or 1 (overall hallucinations)
            # For Inherited (label_filter=0): count where label is 0
            # For Epistemic (label_filter=1): count where label is 1
            if label_filter is None:
                # Overall hallucinations: suspicion_label ∈ {0,1} → hallucination (1), suspicion_label = 2 → no hallucination (0)
                is_confabulation = 1 if suspicion_label in [0, 1] else 0
                case_results.append(is_confabulation)
            else:
                # Specific label filter: count all answers, mark 1 if matches label_filter
                is_confabulation = 1 if suspicion_label == label_filter else 0
                case_results.append(is_confabulation)
        
        if case_results:
            # Calculate case-level average
            case_avg = sum(case_results) / len(case_results)
            cases_dict[case_id] = case_avg
    
    if not cases_dict:
        return None
    
    # Calculate overall average confabulation rate
    case_averages = list(cases_dict.values())
    overall_avg = sum(case_averages) / len(case_averages) if case_averages else 0.0
    
    # Bootstrap CI on case-level averages
    confabulation_ci_lower, confabulation_ci_upper = bootstrap_ci(
        case_averages,
        n_bootstrap=1000
    )
    
    # Calculate total runs and confabulations for reporting
    total_runs = sum(len(case_data.get("answers", [])) for case_data in data.values())
    total_confabulations = 0
    for case_data in data.values():
        for answer in case_data.get("answers", []):
            suspicion_label = _get_suspicion_label(answer)
            if suspicion_label is None:
                continue
            if label_filter is None:
                if suspicion_label in [0, 1]:
                    total_confabulations += 1
            else:
                if suspicion_label == label_filter:
                    total_confabulations += 1
    
    return {
        "total_cases": len(cases_dict),
        "total_runs": total_runs,
        "confabulations": total_confabulations,
        "confabulation_rate": overall_avg,
        "ci_lower": confabulation_ci_lower,
        "ci_upper": confabulation_ci_upper
    }


def _collect_metrics(results_dir: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """
    Collect metrics from all model directories.
    
    Returns:
        Dictionary keyed by (dataset_key, model_display, condition) -> metrics dict
    """
    metrics_map = {}
    
    for model_dir in _discover_model_dirs(results_dir):
        dataset_key = _dataset_from_dirname(model_dir.name)
        if dataset_key is None:
            continue
        
        # Try to determine model name from directory or summary file
        model_display = _normalize_model_name(model_dir.name)
        
        # Check summary file for model name
        summary_path = model_dir / "experiment_summary.json"
        if summary_path.exists():
            try:
                summary = _safe_read_json(summary_path)
                model_tested = summary.get("model_tested", "")
                if model_tested:
                    model_display = _normalize_model_name(model_tested)
            except:
                pass
        
        # Calculate metrics for each condition
        for prompt_name, _prompt_key, settings in PROMPTS:
            for setting_label, condition_alternatives in settings:
                condition_key = condition_alternatives[0]
                metrics = _calculate_metrics_from_json(model_dir, condition_alternatives)
                if metrics:
                    metrics_map[(dataset_key, model_display, condition_key)] = metrics
    
    return metrics_map


def _build_rows(
    metrics_map_overall: Dict[Tuple[str, str, str], Dict[str, Any]],
    metrics_map_label0: Dict[Tuple[str, str, str], Dict[str, Any]],
    metrics_map_label1: Dict[Tuple[str, str, str], Dict[str, Any]]
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Build table rows from metrics maps for Overall, Inherited (0), and Epistemic (1) confabulations.
    
    Table structure matches user requirements:
    - Prompt (dataset) as section header
    - Setting (Default, Default + Mitigation, Temp 0) as sub-section
    - Three confabulation types per setting
    
    Args:
        metrics_map_overall: Dictionary for overall confabulations (0 and 1)
        metrics_map_label0: Dictionary for inherited confabulations (0)
        metrics_map_label1: Dictionary for epistemic confabulations (1)
        
    Returns:
        Tuple of (rows, model_columns_used)
    """
    # Determine which model columns we have
    all_models = set()
    for m in [metrics_map_overall, metrics_map_label0, metrics_map_label1]:
        all_models.update({model for (_, model, _) in m.keys()})
    
    seen_models = sorted(all_models)
    model_cols: List[str] = []
    for m in MODEL_COL_ORDER:
        if m in seen_models:
            model_cols.append(m)
    for m in seen_models:
        if m not in model_cols:
            model_cols.append(m)
    
    rows: List[Dict[str, str]] = []
    
    # Confabulation type labels
    confab_types = [
        ("Overall confabulations", metrics_map_overall),
        ("Inherited confabulations", metrics_map_label0),
        ("Epistemic confabulations", metrics_map_label1),
    ]
    
    for prompt_name, _prompt_key, settings in PROMPTS:
        for dataset_label, dataset_key in DATASETS:
            prompt_label = f"{prompt_name} ({dataset_label})"
            is_first_prompt_row = True
            
            for setting_label, condition_alternatives in settings:
                # Use first condition alternative as key
                condition_key = condition_alternatives[0]
                is_first_setting_row = True
                
                for confab_label, metrics_map in confab_types:
                    row: Dict[str, str] = {
                        "Prompt": prompt_label if is_first_prompt_row and is_first_setting_row else "",
                        "Setting": setting_label if is_first_setting_row else "",
                        "Confabulation Type": confab_label,
                    }
                    
                    for model in model_cols:
                        metrics = metrics_map.get((dataset_key, model, condition_key))
                        cell = ""
                        if metrics:
                            try:
                                cell = _format_rate_ci(
                                    metrics["confabulation_rate"],
                                    metrics["ci_lower"],
                                    metrics["ci_upper"],
                                )
                            except Exception:
                                cell = ""
                        row[model] = cell
                    rows.append(row)
                    is_first_setting_row = False
                    is_first_prompt_row = False
    
    return rows, model_cols


def _rows_to_markdown(rows: List[Dict[str, str]], columns: List[str]) -> str:
    """
    Minimal Markdown table renderer (no external deps).
    """
    def esc(v: str) -> str:
        return (v or "").replace("\n", " ").replace("|", "\\|")

    # column widths
    widths = {c: len(c) for c in columns}
    for r in rows:
        for c in columns:
            widths[c] = max(widths[c], len(esc(str(r.get(c, "")))))

    def fmt_row(vals: List[str]) -> str:
        padded = [f" {esc(v):<{widths[c]}} " for v, c in zip(vals, columns)]
        return "|" + "|".join(padded) + "|"

    header = fmt_row(columns)
    sep = "|" + "|".join([f" {'-' * widths[c]} " for c in columns]) + "|"
    body = "\n".join(fmt_row([str(r.get(c, "")) for c in columns]) for r in rows)
    return "\n".join([header, sep, body])


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, str]], columns: List[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def _write_latex(path: Path, rows: List[Dict[str, str]], columns: List[str], footnote: str = "") -> None:
    """
    Prefer pandas for LaTeX export (nicer formatting); fall back to a simple tabular.
    """
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows, columns=columns)
        # Use booktabs for professional look
        latex = df.to_latex(index=False, escape=True, column_format="l" * len(columns))
        if footnote:
            # Simple way to append footnote after \end{tabular}
            latex = latex.replace("\\end{tabular}", "\\end{tabular}\n\n\\medskip\n\\noindent\n" + footnote.replace("%", "\\%"))
        _write_text(path, latex)
        return
    except Exception:
        pass

    # Fallback: basic LaTeX tabular
    col_spec = "l" * len(columns)
    lines = []
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\hline")
    lines.append(" & ".join(columns) + " \\\\")
    lines.append("\\hline")
    for r in rows:
        vals = [str(r.get(c, "")).replace("&", "\\&").replace("%", "\\%") for c in columns]
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    if footnote:
        lines.append("\n\n" + footnote.replace("%", "\\%"))
    _write_text(path, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hallucination-rate tables from JSON result files.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing per-model subfolders with JSON result files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for generated tables",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap sampling",
    )
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    print("Collecting metrics from JSON files...")
    
    # Discover all model directories
    model_dirs = _discover_model_dirs(results_dir)
    print(f"Found {len(model_dirs)} model directories:")
    for md in model_dirs:
        print(f"  - {md.name}")
    
    # Collect metrics for all three confabulation types
    metrics_map_overall = {}
    metrics_map_label0 = {}
    metrics_map_label1 = {}
    
    for model_dir in model_dirs:
        dataset_key = _dataset_from_dirname(model_dir.name)
        if dataset_key is None:
            print(f"  Warning: Skipping {model_dir.name} - could not determine dataset")
            continue
        
        # Determine model name
        model_display = _normalize_model_name(model_dir.name)
        summary_path = model_dir / "experiment_summary.json"
        if summary_path.exists():
            try:
                summary = _safe_read_json(summary_path)
                model_tested = summary.get("model_tested", "")
                if model_tested:
                    model_display = _normalize_model_name(model_tested)
            except Exception as e:
                print(f"  Warning: Could not read summary from {model_dir.name}: {e}")
        
        print(f"  Processing {model_dir.name} -> {dataset_key}/{model_display}")
        
        # Calculate metrics for each condition and each label filter
        for prompt_name, _prompt_key, settings in PROMPTS:
            for setting_label, condition_alternatives in settings:
                # Use first condition alternative as the key for metrics_map
                condition_key = condition_alternatives[0]
                
                # Overall (0 and 1)
                metrics_overall = _calculate_metrics_from_json(model_dir, condition_alternatives, label_filter=None)
                if metrics_overall:
                    metrics_map_overall[(dataset_key, model_display, condition_key)] = metrics_overall
                
                # Inherited (0)
                metrics_0 = _calculate_metrics_from_json(model_dir, condition_alternatives, label_filter=0)
                if metrics_0:
                    metrics_map_label0[(dataset_key, model_display, condition_key)] = metrics_0
                
                # Epistemic (1)
                metrics_1 = _calculate_metrics_from_json(model_dir, condition_alternatives, label_filter=1)
                if metrics_1:
                    metrics_map_label1[(dataset_key, model_display, condition_key)] = metrics_1

    # Build combined table with all three confabulation types
    print("\nGenerating combined table with Overall, Inherited, and Epistemic confabulations...")
    
    rows, model_cols = _build_rows(metrics_map_overall, metrics_map_label0, metrics_map_label1)
    columns = ["Prompt", "Setting", "Confabulation Type"] + model_cols

    footnote = (
        "\n\nRates reported as mean % (95% Confidence Interval). "
        "All the reported results were obtained from a bootstrapping sampling of size 1000 with 95% Confidence Interval (95% CI). "
        "Mean is calculated from case-level averages across 3 runs."
    )

    md = _rows_to_markdown(rows, columns)
    md += footnote

    # Write outputs
    base_name = "confabulation_rates_table"
    _write_text(out_dir / f"{base_name}.md", md + "\n")
    _write_csv(out_dir / f"{base_name}.csv", rows, columns)
    _write_latex(out_dir / f"{base_name}.tex", rows, columns, footnote=footnote)

    print(f"Saved: {base_name}.md, {base_name}.csv, {base_name}.tex")

    print("\n" + "="*60)
    print("Table generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()