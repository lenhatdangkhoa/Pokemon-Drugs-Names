"""
Metrics calculation for Pokemon experiment results.
"""

import random
from typing import Dict, List, Tuple

import numpy as np

from src.constants import CONDITIONS


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


def calculate_metrics(results: List[Dict], bootstrap_size: int = 1000) -> Dict:
    """
    Calculate hallucination rates per condition using LLM-as-a-Judge results.
    
    For each condition:
    1. Calculate case-level averages (average hallucination rate across 3 runs per case)
    2. Calculate overall average from case-level averages (NOT from bootstrap)
    3. Bootstrap CI on case-level averages (n=1000)
    
    Args:
        results: List of result dictionaries with case_id, condition, suspicion_label, run_number
        bootstrap_size: Number of bootstrap samples for CI calculation
        
    Returns:
        Dictionary with metrics including CI for each condition
    """
    metrics = {}
    
    for condition in CONDITIONS:
        condition_results = [r for r in results if r["condition"] == condition]
        if not condition_results:
            continue
        
        # Group results by case_id to calculate case-level averages
        cases_dict = {}
        for r in condition_results:
            case_id = r["case_id"]
            if case_id not in cases_dict:
                cases_dict[case_id] = []
            # Store hallucination indicator: 1 = hallucination, 0 = no hallucination
            # suspicion_label: 0 = hallucination, 1 or 2 = no hallucination
            is_hallucination = 1 if r["suspicion_label"] == 0 else 0
            cases_dict[case_id].append(is_hallucination)
        
        # Calculate case-level averages (average across runs per case)
        case_averages = []
        for case_id, run_results in cases_dict.items():
            # Average hallucination rate for this case across all runs
            # run_results contains 1s (hallucination) and 0s (no hallucination)
            # So sum/len gives proportion of hallucinations
            case_avg = sum(run_results) / len(run_results) if len(run_results) > 0 else 0.0
            case_averages.append(case_avg)
        
        # Calculate overall average hallucination rate (NOT from bootstrap)
        # This is the mean of case-level averages
        overall_avg = sum(case_averages) / len(case_averages) if len(case_averages) > 0 else 0.0
        
        # Bootstrap CI on case-level averages
        # Convert case averages to list for bootstrapping (they're already rates 0-1)
        hallucination_ci_lower, hallucination_ci_upper = bootstrap_ci(
            case_averages, 
            n_bootstrap=bootstrap_size
        )
        
        # Calculate total runs and hallucinations for reporting
        total_runs = len(condition_results)
        total_hallucinations = sum(1 for r in condition_results if r["suspicion_label"] == 0)
        total_cases = len(cases_dict)
        
        metrics[condition] = {
            "total_cases": total_cases,
            "total_runs": total_runs,
            "hallucinations": total_hallucinations,
            "hallucination_rate": overall_avg,  # Average across cases (not from bootstrap)
            "ci_lower": hallucination_ci_lower,
            "ci_upper": hallucination_ci_upper
        }
    
    return metrics
