#!/usr/bin/env python3
"""
Analyze evaluation results to identify patterns and insights.

This script provides additional analysis beyond basic accuracy metrics,
including position bias analysis, medication-specific patterns, and 
statistical insights.
"""

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def load_evaluation(filepath: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_test_cases(filepath: str) -> List[Dict]:
    """Load test cases from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_position_bias(test_cases: List[Dict], evaluation: Dict) -> Dict:
    """
    Analyze if fabricated medication position affects detection rate.
    
    Returns:
        Dictionary with position-based statistics
    """
    position_stats = defaultdict(lambda: {'total': 0, 'detected': 0})
    
    results_by_id = {r['test_case_id']: r for r in evaluation['individual_results']}
    
    for test_case in test_cases:
        position = test_case['fabricated_position']
        test_id = test_case['id']
        
        position_stats[position]['total'] += 1
        
        if results_by_id[test_id]['fabricated_detected']:
            position_stats[position]['detected'] += 1
    
    # Calculate detection rates
    position_analysis = {}
    for pos, stats in position_stats.items():
        rate = stats['detected'] / stats['total'] if stats['total'] > 0 else 0
        position_analysis[pos] = {
            'position': pos,
            'total_cases': stats['total'],
            'detected': stats['detected'],
            'detection_rate': rate
        }
    
    return dict(sorted(position_analysis.items()))


def analyze_fabricated_medications(test_cases: List[Dict], evaluation: Dict) -> Dict:
    """
    Analyze which fabricated medications are detected most/least often.
    
    Returns:
        Dictionary with medication-specific statistics
    """
    med_stats = defaultdict(lambda: {'total': 0, 'detected': 0})
    
    results_by_id = {r['test_case_id']: r for r in evaluation['individual_results']}
    
    for test_case in test_cases:
        fabricated = test_case['fabricated_medication']
        test_id = test_case['id']
        
        med_stats[fabricated]['total'] += 1
        
        if results_by_id[test_id]['fabricated_detected']:
            med_stats[fabricated]['detected'] += 1
    
    # Calculate detection rates
    med_analysis = {}
    for med, stats in med_stats.items():
        rate = stats['detected'] / stats['total'] if stats['total'] > 0 else 0
        med_analysis[med] = {
            'medication': med,
            'appearances': stats['total'],
            'detected': stats['detected'],
            'detection_rate': rate
        }
    
    # Sort by detection rate (ascending) to find hardest to detect
    sorted_meds = sorted(med_analysis.values(), key=lambda x: x['detection_rate'])
    
    return {
        'all_medications': med_analysis,
        'hardest_to_detect': sorted_meds[:5],
        'easiest_to_detect': sorted_meds[-5:]
    }


def print_position_analysis(position_analysis: Dict):
    """Print position bias analysis results."""
    print("\n" + "="*60)
    print("POSITION BIAS ANALYSIS")
    print("="*60)
    print(f"{'Position':<10} {'Total':<10} {'Detected':<10} {'Rate':<10}")
    print("-"*60)
    
    for pos, stats in position_analysis.items():
        print(f"{pos:<10} {stats['total_cases']:<10} "
              f"{stats['detected']:<10} {stats['detection_rate']:.2%}")
    
    print("="*60 + "\n")


def print_medication_analysis(med_analysis: Dict):
    """Print medication-specific analysis results."""
    print("\n" + "="*60)
    print("HARDEST TO DETECT FABRICATED MEDICATIONS")
    print("="*60)
    
    for med in med_analysis['hardest_to_detect']:
        print(f"{med['medication']:<15} - Detected {med['detected']}/{med['appearances']} "
              f"times ({med['detection_rate']:.2%})")
    
    print("\n" + "="*60)
    print("EASIEST TO DETECT FABRICATED MEDICATIONS")
    print("="*60)
    
    for med in med_analysis['easiest_to_detect']:
        print(f"{med['medication']:<15} - Detected {med['detected']}/{med['appearances']} "
              f"times ({med['detection_rate']:.2%})")
    
    print("="*60 + "\n")


def generate_insights(
    evaluation: Dict,
    position_analysis: Dict,
    med_analysis: Dict
) -> List[str]:
    """Generate actionable insights from the analysis."""
    insights = []
    
    # Overall performance insight
    accuracy = evaluation['accuracy']
    if accuracy >= 0.9:
        insights.append("✓ Excellent performance: LLM demonstrates strong ability to identify fabricated medications")
    elif accuracy >= 0.7:
        insights.append("• Good performance: LLM shows reasonable detection capability with room for improvement")
    else:
        insights.append("⚠ Concerning performance: LLM struggles to reliably identify fabricated medications")
    
    # False positive insight
    false_positive_rate = evaluation['avg_false_positives_per_case']
    if false_positive_rate > 0.5:
        insights.append("⚠ High false positive rate: LLM frequently misidentifies real medications as fabricated")
    elif false_positive_rate > 0.1:
        insights.append("• Moderate false positives: Some real medications incorrectly flagged")
    else:
        insights.append("✓ Low false positives: LLM rarely misidentifies real medications")
    
    # Position bias insight
    if position_analysis:
        rates = [stats['detection_rate'] for stats in position_analysis.values()]
        if max(rates) - min(rates) > 0.2:
            insights.append("⚠ Position bias detected: Detection rate varies significantly based on list position")
        else:
            insights.append("✓ No significant position bias: Detection rate consistent across positions")
    
    return insights


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze evaluation results for patterns and insights"
    )
    parser.add_argument(
        "--evaluation",
        type=str,
        required=True,
        help="Path to evaluation results JSON file"
    )
    parser.add_argument(
        "--test-cases",
        type=str,
        required=True,
        help="Path to test cases JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save analysis results as JSON"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading evaluation from {args.evaluation}...")
    evaluation = load_evaluation(args.evaluation)
    
    print(f"Loading test cases from {args.test_cases}...")
    test_cases = load_test_cases(args.test_cases)
    
    # Perform analyses
    print("Analyzing position bias...")
    position_analysis = analyze_position_bias(test_cases, evaluation)
    
    print("Analyzing fabricated medications...")
    med_analysis = analyze_fabricated_medications(test_cases, evaluation)
    
    # Generate insights
    insights = generate_insights(evaluation, position_analysis, med_analysis)
    
    # Display results
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Test Cases Analyzed: {evaluation['total_test_cases']}")
    print(f"Overall Accuracy: {evaluation['accuracy']:.2%}")
    print(f"False Positive Rate: {evaluation['avg_false_positives_per_case']:.2f} per case")
    print("="*60)
    
    print_position_analysis(position_analysis)
    print_medication_analysis(med_analysis)
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    for insight in insights:
        print(f"  {insight}")
    print("="*60 + "\n")
    
    # Save if requested
    if args.output:
        analysis_results = {
            'summary': {
                'total_cases': evaluation['total_test_cases'],
                'accuracy': evaluation['accuracy'],
                'false_positive_rate': evaluation['avg_false_positives_per_case']
            },
            'position_bias': position_analysis,
            'medication_analysis': med_analysis,
            'insights': insights
        }
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"Analysis results saved to {output_path}")


if __name__ == "__main__":
    main()
