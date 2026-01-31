"""
Results formatting and display utilities for Pokemon experiment.
"""

from typing import Dict


def display_results_table(metrics: Dict, model_name: str) -> None:
    """
    Display experiment results in formatted table.
    
    Args:
        metrics: Dictionary with metrics per condition
        model_name: Name of the model tested
    """
    from src.constants import CONDITIONS
    
    dosing_conditions = ["default", "mitigation", "temp0"]
    indication_conditions = ["medication_indication", "medication_indication_mitigation", "medication_indication_temp0"]
    
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS")
    print("="*80)
    print("(Hallucination detection using LLM-as-a-Judge)")
    print("(95% Confidence Interval from bootstrapping, n=1000)")
    print("="*80)
    
    # Display results in table format
    print("\n" + "-"*80)
    print(f"{'Condition':<40} {'Hallucination Rate':<20} {'95% CI':<20}")
    print("-"*80)
    
    print("\nDrug Dosing Prompt:")
    for condition in dosing_conditions:
        if condition in metrics:
            stats = metrics[condition]
            condition_display = {
                "default": "Default",
                "mitigation": "Default + Mitigation",
                "temp0": "Temp 0"
            }.get(condition, condition)
            rate_str = f"{stats['hallucination_rate']:.2%}"
            ci_str = f"[{stats['ci_lower']:.2%}, {stats['ci_upper']:.2%}]"
            print(f"  {condition_display:<38} {rate_str:<20} {ci_str:<20}")
    
    print("\nDrug Indication Prompt:")
    for condition in indication_conditions:
        if condition in metrics:
            stats = metrics[condition]
            condition_display = {
                "medication_indication": "Default",
                "medication_indication_mitigation": "Default + Mitigation",
                "medication_indication_temp0": "Temp 0"
            }.get(condition, condition)
            rate_str = f"{stats['hallucination_rate']:.2%}"
            ci_str = f"[{stats['ci_lower']:.2%}, {stats['ci_upper']:.2%}]"
            print(f"  {condition_display:<38} {rate_str:<20} {ci_str:<20}")
    
    print("\n" + "-"*80)
    print(f"\nDetailed Statistics:")
    for condition, stats in metrics.items():
        print(f"\n{condition.upper()}:")
        print(f"  Total cases: {stats['total_cases']}")
        print(f"  Total runs: {stats['total_runs']}")
        print(f"  Hallucinations: {stats['hallucinations']}")
        print(f"  Average hallucination rate: {stats['hallucination_rate']*100:.2f}%")
        print(f"  95% CI: [{stats['ci_lower']*100:.2f}, {stats['ci_upper']*100:.2f}]")
    
    # Print final formatted table for GPT-4o-mini model results
    print("\n" + "="*90)
    print(f"FINAL RESULTS TABLE - {model_name.upper()} MODEL")
    print("="*90)
    print("\nHallucination Rates with 95% Confidence Intervals")
    print("(All results obtained from bootstrapping sampling of size 1000 with 95% CI)\n")
    
    # Create formatted table with proper alignment
    # Format: 86.92% [80.25, 92.73]
    col_widths = [45, 43]
    separator = "+" + "-"*col_widths[0] + "+" + "-"*col_widths[1] + "+"
    
    print(separator)
    print(f"| {'Condition':<{col_widths[0]-1}}| {'Hallucination Rate (95% CI)':<{col_widths[1]-1}}|")
    print(separator)
    
    # Drug Dosing Prompt section
    print(f"| {'Drug Dosing Prompt':<{col_widths[0]-1}}| {'':<{col_widths[1]-1}}|")
    print(separator)
    
    for condition in dosing_conditions:
        if condition in metrics:
            stats = metrics[condition]
            condition_display = {
                "default": "  Default",
                "mitigation": "  Default + Mitigation",
                "temp0": "  Temp 0"
            }.get(condition, f"  {condition}")
            # Format: 86.92% [80.25, 92.73]
            rate_str = f"{stats['hallucination_rate']*100:.2f}%"
            ci_str = f"[{stats['ci_lower']*100:.2f}, {stats['ci_upper']*100:.2f}]"
            combined_str = f"{rate_str} {ci_str}"
            print(f"| {condition_display:<{col_widths[0]-1}}| {combined_str:<{col_widths[1]-1}}|")
    
    print(separator)
    
    # Drug Indication Prompt section
    print(f"| {'Drug Indication Prompt':<{col_widths[0]-1}}| {'':<{col_widths[1]-1}}|")
    print(separator)
    
    for condition in indication_conditions:
        if condition in metrics:
            stats = metrics[condition]
            condition_display = {
                "medication_indication": "  Default",
                "medication_indication_mitigation": "  Default + Mitigation",
                "medication_indication_temp0": "  Temp 0"
            }.get(condition, f"  {condition}")
            # Format: 86.92% [80.25, 92.73]
            rate_str = f"{stats['hallucination_rate']*100:.2f}%"
            ci_str = f"[{stats['ci_lower']*100:.2f}, {stats['ci_upper']*100:.2f}]"
            combined_str = f"{rate_str} {ci_str}"
            print(f"| {condition_display:<{col_widths[0]-1}}| {combined_str:<{col_widths[1]-1}}|")
    
    print(separator)
    print("\nNote: Average hallucination rate is calculated from case-level averages (across 3 runs).")
    print("      95% Confidence Intervals were obtained from bootstrapping sampling of size 1000.\n")
    
    print("="*90)
