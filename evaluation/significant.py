"""
Paired-Permutation Significance Tests for Pokemon Probing Experiments
======================================================================

python evaluation/significant.py --table-generic-mitigation

This module implements exact and approximate paired-permutation significance tests
following the framework of Zmigrod et al. (NAACL 2022) for comparing LLM outputs
across experimental conditions.

Experimental Comparisons
------------------------
We compare conditions for each (dataset_type × prompt_type × model):
  - (Default + No Mitigation) vs (Default + Mitigation)
  - (Default + No Mitigation) vs Temp 0
  - (Default + Mitigation) vs Temp 0

Data sources (per model directory under `results/`):
  - Drug Indication prompt:
      `medication_indication.json` (Default + No Mitigation)
      `medication_indication_mitigation.json` (Default + Mitigation)
      `medication_indication_temp0.json` (Temp 0)
  - Drug Dosing prompt:
      `default.json` (Default + No Mitigation)
      `mitigation.json` (Default + Mitigation)
      `temp0.json` (Temp 0)

Statistical Framework: Paired Permutation Test
----------------------------------------------
Let (x_n, y_n) for n ∈ {1, ..., N} be paired outcomes from two systems A and B
on the same N test examples. The paired permutation test assesses whether the
observed difference is significant under the null hypothesis H₀ that both
systems are exchangeable (i.e., the distribution of (x_n, y_n) equals that
of (y_n, x_n) for all n).

**Null Hypothesis (H₀):**
  For each example n, the assignment to system A vs B is arbitrary; swapping
  the labels does not change the joint distribution.

**Test Statistic:**
  We use the mean difference: T = (1/N) Σ_n (x_n - y_n)

**Permutation Distribution:**
  Under H₀, each pair (x_n, y_n) can be swapped with probability 0.5 independently.
  Equivalently, let s_n ∈ {+1, -1} be independent Rademacher random variables;
  the null distribution of T is the distribution of T_π = (1/N) Σ_n s_n (x_n - y_n).

**Two-Sided P-value:**
  p = P_{H₀}(|T_π| ≥ |T_obs|)

Implementation Details
----------------------

1. **Binary Outcomes (McNemar's Exact Test Equivalence):**
   When x_n, y_n ∈ {0, 1}, the test reduces to counting discordant pairs:
     - wins = #{n : x_n = 1, y_n = 0}
     - losses = #{n : x_n = 0, y_n = 1}
     - k = wins + losses (total discordant pairs)
   
   Under H₀, the number of wins W ~ Binomial(k, 0.5).
   The two-sided p-value is:
     p = P(|W - k/2| ≥ |wins - k/2|) = P(W ≤ ⌊(k-t)/2⌋) + P(W ≥ ⌈(k+t)/2⌉)
   where t = |wins - losses|.
   
   This is mathematically equivalent to McNemar's exact test and is computed
   via `scipy.stats.binom` without any Monte Carlo sampling.

2. **Integer-Valued Deltas (Exact DP Algorithm):**
   When aggregating R runs per example, we have:
     u_n = Σ_{r=1}^{R} x_{n,r} ∈ {0, 1, ..., R}  (count for system A)
     v_n = Σ_{r=1}^{R} y_{n,r} ∈ {0, 1, ..., R}  (count for system B)
     δ_n = u_n - v_n ∈ {-R, ..., R}              (integer delta)
   
   The test statistic is S = Σ_n δ_n, and under H₀:
     S_π = Σ_n s_n δ_n  where s_n ∈ {+1, -1} uniformly
   
   We compute the exact PMF of S_π using dynamic programming (DP) via
   convolution. Starting with PMF = δ_0 (point mass at 0), for each δ_n ≠ 0:
     PMF ← 0.5 × (shift(PMF, +δ_n) + shift(PMF, -δ_n))
   
   The two-sided p-value is: p = Σ_{|s| ≥ |S_obs|} PMF(s)
   
   This matches Algorithm 1 in Zmigrod et al. (2022) for additively
   decomposable integer statistics.

3. **Monte Carlo Approximation:**
   For non-integer or large-scale data, we use MC sampling:
     - Draw M permutations by randomly swapping each pair with p = 0.5
     - Compute the test statistic T_m for each permutation m ∈ {1, ..., M}
     - Estimate p ≈ (1/M) Σ_m 𝟙(|T_m| ≥ |T_obs|)
   
   Default: M = 25,000 samples (as in testSig3.py / rycolab/paired-perm-test).

Handling Multiple Runs Per Example
----------------------------------
Each example may have R independent runs (repetitions). We aggregate using
per-example counts (recommended):

  u_n = number of hallucinations (suspicion_label ∈ {0,1}) for system A on example n
  v_n = number of hallucinations (suspicion_label ∈ {0,1}) for system B on example n

This preserves the paired structure: example n under condition A vs condition B.
The integer delta δ_n = u_n - v_n is then used with the exact DP algorithm.

This approach:
  - Respects the paired nature of the test (same N examples, different conditions)
  - Avoids inflating significance by treating runs as independent samples
  - Is consistent with the NAACL 2022 framework for structured statistics

Mathematical Consistency Check
------------------------------
The three implementations are consistent:
  - Binary test (k discordant pairs) → exact via Binom(k, 0.5) tails
  - Integer DP (N examples with δ_n ∈ ℤ) → exact via convolution
  - MC (arbitrary outcomes) → approximate via random sign flips

For the special case of binary outcomes with R=1 run:
  - Binary test uses k = wins + losses discordant pairs
  - Integer DP uses δ_n ∈ {-1, 0, +1} with exact PMF
  - Both yield identical p-values (verified numerically)

References
----------
[1] Zmigrod, R., Vieira, T., & Cotterell, R. (2022).
    "Exact Paired-Permutation Testing for Structured Test Statistics."
    Proceedings of NAACL 2022.
    https://aclanthology.org/2022.naacl-main.360/
    arXiv: https://arxiv.org/abs/2205.01416

[2] rycolab/paired-perm-test: Reference implementation
    https://github.com/rycolab/paired-perm-test

[3] McNemar, Q. (1947). "Note on the sampling error of the difference
    between correlated proportions or percentages." Psychometrika, 12(2).

Usage
-----
python ./evaluation/significant.py

  --runs N              Use up to N runs per example (default: 3)
  --run-aggregation     'counts' (exact DP), 'mean' (MC), or 'flatten' (MC)
  --table4              Generate pairwise model comparison table
  --table-generic-mitigation  Generate generic dataset + mitigation table
  --alpha 0.05          Significance threshold


python evaluation/significant.py --table-generic-mitigation
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import binom


RESULTS_DIR = Path("./results")
OUT_DIR = Path("./evaluation/stat")


@dataclass(frozen=True)
class Comparison:
    key: str
    label: str
    a_name: str
    b_name: str


COMPARISONS: List[Comparison] = [
    Comparison(
        key="default_vs_mitigation",
        label="(Default + No Mitigation) vs. (Default + Mitigation)",
        a_name="default",
        b_name="mitigation",
    ),
    Comparison(
        key="default_vs_temp0",
        label="(Default + No Mitigation) vs. Temp 0",
        a_name="default",
        b_name="temp0",
    ),
    Comparison(
        key="mitigation_vs_temp0",
        label="(Default + Mitigation) vs. Temp 0",
        a_name="mitigation",
        b_name="temp0",
    ),
]


PROMPTS = {
    # prompt_type -> {condition_name -> filename}
    "drug_indication": {
        "default": "medication_indication.json",
        "mitigation": "medication_indication_mitigation.json",
        "temp0": "medication_indication_temp0.json",
    },
    "drug_dosing": {
        "default": "default.json",
        "mitigation": "mitigation.json",
        "temp0": "temp0.json",
    },
}


SECTION_TITLES = {
    ("generic", "drug_indication"): "Generic Dataset and Drug Indication Prompt",
    ("generic", "drug_dosing"): "Generic Dataset and Drug Dosing Prompt",
    ("brand", "drug_indication"): "Brand Dataset and Drug Indication Prompt",
    ("brand", "drug_dosing"): "Brand Dataset and Drug Dosing Prompt",
}

# Display names for Table 4 style output
MODEL_DISPLAY = {
    "llama": "Llama-3.3-70B-Instruct",
    "llama3": "Llama-3.3-70B-Instruct",
    "gemma": "Gemma-3-27B-IT",
    "gpt4o": "GPT-4o-mini",
    "gpt_4o": "GPT-4o-mini",
    "qwen": "Qwen3",
}


def _safe_float(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _load_json(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sorted_item_keys(d: Dict[str, dict]) -> List[str]:
    # Keys are strings of ints ("1", "2", ...) in your files.
    def to_int(k: str) -> int:
        try:
            return int(k)
        except Exception:
            return 10**18

    return sorted(d.keys(), key=to_int)


def extract_suspicion_detected_matrix(data: Dict[str, dict]) -> np.ndarray:
    """
    Returns a matrix shaped (n_items, n_runs_available) of {0,1} hallucination values,
    where each run is an entry in the `answers` list.
    
    Uses overall hallucinations: suspicion_label ∈ {0,1} → hallucination (1),
    suspicion_label = 2 → no hallucination (0).
    """
    keys = _sorted_item_keys(data)
    rows: List[List[int]] = []
    min_runs: Optional[int] = None

    for k in keys:
        obj = data.get(k, {})
        answers = obj.get("answers", [])
        run_vals: List[int] = []
        for ans in answers:
            # Extract suspicion_label: 0 or 1 = hallucination (1), 2 = no hallucination (0)
            # Default to 2 (no hallucination) if missing
            suspicion_label = ans.get("suspicion_label", 2)
            # suspicion_label ∈ {0,1} → hallucination (1), suspicion_label = 2 → no hallucination (0)
            run_vals.append(1 if suspicion_label in [0, 1] else 0)
        if min_runs is None:
            min_runs = len(run_vals)
        else:
            min_runs = min(min_runs, len(run_vals))
        rows.append(run_vals)

    if min_runs is None or min_runs == 0:
        return np.zeros((0, 0), dtype=int)

    # Truncate to the minimum run count across items to keep runs aligned.
    mat = np.array([r[:min_runs] for r in rows], dtype=int)
    return mat


def paired_permutation_pvalue_binary(x: np.ndarray, y: np.ndarray, two_sided: bool = True) -> float:
    """
    Exact paired permutation p-value for binary paired outcomes (McNemar's exact test).

    Mathematical Framework (following Zmigrod et al., NAACL 2022):
    ---------------------------------------------------------------
    For paired binary outcomes (x_n, y_n) ∈ {0,1}², only discordant pairs matter:
      - wins   = #{n : x_n = 1, y_n = 0}  (system A wins)
      - losses = #{n : x_n = 0, y_n = 1}  (system B wins)
      - k = wins + losses                  (total discordant pairs)

    Under H₀ (exchangeability), each discordant pair is equally likely to favor
    either system. Thus, the number of wins W ~ Binomial(k, 0.5).

    Test statistic: T = wins - losses (observed difference)
    Null distribution: T_π = 2W - k where W ~ Binom(k, 0.5)

    Two-sided p-value:
      p = P(|2W - k| ≥ |T|) = P(W ≤ ⌊(k-|T|)/2⌋) + P(W ≥ ⌈(k+|T|)/2⌉)

    This is mathematically equivalent to McNemar's exact test (McNemar, 1947).

    Parameters
    ----------
    x : np.ndarray
        Binary outcomes for system A, shape (N,)
    y : np.ndarray
        Binary outcomes for system B, shape (N,)
    two_sided : bool
        If True, compute two-sided p-value (default)

    Returns
    -------
    float
        Exact two-sided p-value in [0, 1]
    """
    x = np.asarray(x).astype(int).ravel()
    y = np.asarray(y).astype(int).ravel()
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x{tuple(x.shape)} vs y{tuple(y.shape)}")
    if x.size == 0:
        return 1.0

    # Ensure binary
    if not (np.isin(x, [0, 1]).all() and np.isin(y, [0, 1]).all()):
        raise ValueError("paired_permutation_pvalue_binary expects x,y in {0,1}.")

    wins = int(np.sum((x == 1) & (y == 0)))
    losses = int(np.sum((x == 0) & (y == 1)))
    k = wins + losses
    if k == 0:
        return 1.0

    t_obs = abs(wins - losses)  # |2*wins - k|
    # Two-sided exact tail: P(|2W - k| >= t_obs), W~Binom(k,0.5)
    # Compute by summing tails on W.
    # Condition |2W-k| >= t <=> W <= floor((k - t)/2) OR W >= ceil((k + t)/2)
    lo = (k - t_obs) // 2
    hi = (k + t_obs + 1) // 2  # ceil((k+t)/2)

    # P(W <= lo) + P(W >= hi)
    p_lo = binom.cdf(lo, k, 0.5)
    p_hi = binom.sf(hi - 1, k, 0.5)  # sf is P(W > hi-1) = P(W >= hi)
    p = float(p_lo + p_hi)

    # Numerical guard
    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0
    return p


def paired_permutation_pvalue_mc(x: np.ndarray, y: np.ndarray, mc: int, seed: int = 0) -> float:
    """
    Monte Carlo approximation of paired permutation p-value for arbitrary outcomes.

    Mathematical Framework (following Zmigrod et al., NAACL 2022):
    ---------------------------------------------------------------
    For paired outcomes (x_n, y_n), the test statistic is T = mean(x - y).

    Under H₀ (exchangeability), we simulate the null distribution by:
      1. For each permutation m ∈ {1, ..., M}:
         - For each pair n, swap (x_n, y_n) with probability 0.5
         - Compute T_m = mean(x_perm - y_perm)
      2. Estimate p-value: p ≈ (1/M) Σ_m 𝟙(|T_m| ≥ |T_obs|)

    This is equivalent to randomly flipping the sign of each difference d_n = x_n - y_n,
    which matches the paired permutation null hypothesis.

    Parameters
    ----------
    x : np.ndarray
        Outcomes for system A, shape (N,)
    y : np.ndarray
        Outcomes for system B, shape (N,)
    mc : int
        Number of Monte Carlo samples (default: 25,000 in rycolab/paired-perm-test)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    float
        Approximate two-sided p-value in [0, 1]
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x{tuple(x.shape)} vs y{tuple(y.shape)}")
    if x.size == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    stat_obs = float(np.mean(x - y))
    # random mask: True => swap
    mask = rng.random((mc, x.size)) < 0.5
    # Vectorized swapping
    x_rep = np.broadcast_to(x, (mc, x.size)).copy()
    y_rep = np.broadcast_to(y, (mc, x.size)).copy()
    x_rep[mask], y_rep[mask] = y_rep[mask], x_rep[mask]
    stats = np.mean(x_rep - y_rep, axis=1)
    p = float(np.mean(np.abs(stats) >= abs(stat_obs)))
    return min(max(p, 0.0), 1.0)


def exact_paired_perm_pvalue_integer_deltas(deltas: np.ndarray) -> float:
    """
    Exact paired-permutation p-value for integer-valued deltas using DP convolution.

    Mathematical Framework (Algorithm 1 in Zmigrod et al., NAACL 2022):
    --------------------------------------------------------------------
    For N examples with integer deltas δ_n = u_n - v_n (e.g., count differences):

    Test statistic: S = Σ_n δ_n (sum of deltas)

    Under H₀ (exchangeability), swapping systems for example n flips the sign of δ_n.
    The null distribution is:
      S_π = Σ_n s_n · δ_n  where s_n ∈ {+1, -1} uniformly and independently

    Exact PMF Computation via Dynamic Programming:
      1. Initialize PMF = δ₀ (point mass at 0)
      2. For each δ_n ≠ 0:
           PMF ← 0.5 × (shift(PMF, +δ_n) + shift(PMF, -δ_n))
         This convolves with the distribution of s_n · δ_n ∈ {-δ_n, +δ_n}
      3. Final PMF gives P(S_π = s) for each integer s ∈ [-Σ|δ_n|, +Σ|δ_n|]

    Two-sided p-value: p = Σ_{|s| ≥ |S_obs|} PMF(s)

    Complexity: O(N · (Σ|δ_n|)) time and O(Σ|δ_n|) space

    Reference: https://arxiv.org/abs/2205.01416

    Parameters
    ----------
    deltas : np.ndarray
        Integer deltas δ_n = u_n - v_n for each example n, shape (N,)

    Returns
    -------
    float
        Exact two-sided p-value in [0, 1]
    """
    deltas = np.asarray(deltas, dtype=int).ravel()
    if deltas.size == 0:
        return 1.0

    s_obs = int(np.sum(deltas))
    t_obs = abs(s_obs)
    abs_sum = int(np.sum(np.abs(deltas)))
    if abs_sum == 0:
        return 1.0

    offset = abs_sum
    pmf = np.zeros(2 * offset + 1, dtype=np.float64)
    pmf[offset] = 1.0

    def shift(arr: np.ndarray, k: int) -> np.ndarray:
        out = np.zeros_like(arr)
        if k == 0:
            out[:] = arr
            return out
        if k > 0:
            out[k:] = arr[:-k]
        else:
            kk = -k
            out[:-kk] = arr[kk:]
        return out

    for d in deltas:
        if d == 0:
            continue
        pmf = 0.5 * (shift(pmf, d) + shift(pmf, -d))

    idx = np.arange(-offset, offset + 1)
    p = float(np.sum(pmf[np.abs(idx) >= t_obs]))
    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0
    return p


def compute_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    mc: int = 0,
    seed: int = 0,
) -> float:
    # Prefer exact binary when possible; otherwise fall back to MC.
    if a.size == 0 or b.size == 0:
        return 1.0
    if np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all():
        return paired_permutation_pvalue_binary(a, b, two_sided=True)
    # Non-binary: use MC paired permutation (run-aggregated means will hit this path).
    if mc <= 0:
        mc = 25_000
    return paired_permutation_pvalue_mc(a, b, mc=mc, seed=seed)


def iter_model_dirs(results_dir: Path) -> Iterable[Tuple[str, str, Path]]:
    """
    Yields (dataset_type, model_name, path).
    dataset_type is 'generic' or 'brand' based on directory prefix.
    """
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("generic_"):
            yield "generic", name[len("generic_") :], child
        elif name.startswith("brand_"):
            yield "brand", name[len("brand_") :], child


def load_prompt_condition_matrix(model_dir: Path, prompt_type: str, condition: str) -> Optional[np.ndarray]:
    """
    Load prompt condition matrix, trying multiple filename alternatives.
    
    For drug_dosing, handles both naming conventions:
    - Standard: default.json, mitigation.json, temp0.json
    - Alternative: drug_dosing.json, drug_dosing_mitigation.json, drug_dosing_temp0.json
    """
    # Primary filename from PROMPTS
    filename = PROMPTS[prompt_type][condition]
    path = model_dir / filename
    
    # If primary doesn't exist, try alternative naming for drug_dosing
    if not path.exists() and prompt_type == "drug_dosing":
        alternative_names = {
            "default": "drug_dosing.json",
            "mitigation": "drug_dosing_mitigation.json",
            "temp0": "drug_dosing_temp0.json",
        }
        if condition in alternative_names:
            alt_filename = alternative_names[condition]
            alt_path = model_dir / alt_filename
            if alt_path.exists():
                path = alt_path
    
    if not path.exists():
        return None
    
    data = _load_json(path)
    return extract_suspicion_detected_matrix(data)


def per_example_counts(a_mat: np.ndarray, b_mat: np.ndarray, max_runs: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Convert (n_items, n_runs) binary matrices into per-example integer counts using aligned runs:

      u_n = sum_r a_{n,r}
      v_n = sum_r b_{n,r}

    Returns (u, v, runs_used).
    """
    if a_mat is None or b_mat is None:
        return np.array([], dtype=int), np.array([], dtype=int), 0
    n = min(a_mat.shape[0], b_mat.shape[0])
    runs_used = min(max_runs, a_mat.shape[1], b_mat.shape[1])
    if n <= 0 or runs_used <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), 0
    u = a_mat[:n, :runs_used].sum(axis=1).astype(int)
    v = b_mat[:n, :runs_used].sum(axis=1).astype(int)
    return u, v, runs_used


def write_pairedperm_input(path: Path, u: Sequence[int], v: Sequence[int]) -> None:
    """
    Write a paired-perm-test compatible CSV file: one entry per line: `u_n,v_n` (integers).
    Repo: https://github.com/rycolab/paired-perm-test
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for uu, vv in zip(u, v):
            f.write(f"{int(uu)},{int(vv)}\n")


def per_example_counts_single(mat: np.ndarray, max_runs: int) -> Tuple[np.ndarray, int]:
    """
    For a single system/condition matrix (n_items, n_runs) of 0/1, compute per-example integer counts (0..R).
    Returns (counts, runs_used).
    """
    if mat is None or mat.size == 0:
        return np.array([], dtype=int), 0
    runs_used = min(max_runs, mat.shape[1])
    if runs_used <= 0:
        return np.array([], dtype=int), 0
    return mat[:, :runs_used].sum(axis=1).astype(int), runs_used


def pairwise_model_pvalue_counts(
    mat_a: np.ndarray, mat_b: np.ndarray, max_runs: int
) -> Tuple[float, float, int, int]:
    """
    Compare two models on the same prompt/dataset/condition using per-example run-counts and exact DP.
    Returns (value, p_value, n_examples, runs_used).
    """
    a_counts, ra = per_example_counts_single(mat_a, max_runs=max_runs)
    b_counts, rb = per_example_counts_single(mat_b, max_runs=max_runs)
    if a_counts.size == 0 or b_counts.size == 0:
        return float("nan"), float("nan"), 0, 0
    n = min(a_counts.size, b_counts.size)
    r = min(ra, rb)
    if n <= 0 or r <= 0:
        return float("nan"), float("nan"), 0, 0
    # If run counts differ, truncate to r runs by scaling is not correct; instead recompute counts with r.
    # We do this by re-summing the original matrices up to r runs.
    a_counts = mat_a[:n, :r].sum(axis=1).astype(int)
    b_counts = mat_b[:n, :r].sum(axis=1).astype(int)
    deltas = (a_counts - b_counts).astype(int)
    p = exact_paired_perm_pvalue_integer_deltas(deltas)
    value = float(np.sum(deltas) / float(n * r))
    return value, p, n, r


def _model_pairs(models: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            pairs.append((models[i], models[j]))
    return pairs


def print_pairwise_table4(rows: List[Tuple[str, str, str]]) -> None:
    """
    rows: (section_title_or_empty, label, p_str)
    """
    print("\n" + "=" * 100)
    print("Table 4. Pairwise Comparisons")
    print()
    print("Method: We compare models on their detection rates (proportion of test cases where")
    print("the model flagged the prompt as suspicious). For each pair of models tested on the")
    print("same examples, we compute the difference in detection rates and test its significance")
    print("using exact paired-permutation tests (Zmigrod et al., NAACL 2022).")
    print()
    print("Interpretation: A significant p-value (p < 0.05) indicates that one model detects")
    print("suspicious prompts significantly more often than the other when evaluated on the same")
    print("test cases. Lower p-values indicate stronger evidence for a difference.")
    print("=" * 100)
    print(f"{'':<2}{'Pairwise Comparison':<60} {'P value':>12}")
    print("-" * 100)
    cur = None
    for section, label, p_str in rows:
        if section and section != cur:
            cur = section
            print(f"\n{section}")
        print(f"  {label:<60} {p_str:>12}")
    print()


def get_generic_mitigation_section_title(prompt_type: str) -> str:
    """
    Returns the section title for generic dataset with mitigation prompt.
    Format matches the image: "Drug Dosing Prompt (using generic drug dataset and mitigation prompt)"
    """
    if prompt_type == "drug_dosing":
        return "Drug Dosing Prompt (using generic drug dataset and mitigation prompt)"
    elif prompt_type == "drug_indication":
        return "Drug Indication Prompt (using generic drug dataset and mitigation prompt)"
    else:
        return f"{prompt_type} (using generic drug dataset and mitigation prompt)"


def print_generic_mitigation_table(rows: List[Tuple[str, str, str]]) -> None:
    """
    Print pairwise comparisons table for generic dataset with mitigation prompt.
    rows: (section_title_or_empty, label, p_str)
    """
    print("\n" + "=" * 100)
    print("Pairwise Model Comparisons (Generic Dataset, Mitigation Prompt)")
    print()
    print("Method: We compare models on their hallucination rates (proportion of test cases where")
    print("suspicion_label ∈ {0,1}, indicating overall hallucinations). For each pair of models")
    print("tested on the same examples, we compute the difference in hallucination rates and")
    print("test its significance using exact paired-permutation tests (Zmigrod et al., NAACL 2022).")
    print()
    print("Interpretation: A significant p-value (p < 0.05) indicates that one model produces")
    print("hallucinations significantly more often than the other when evaluated on the same")
    print("test cases. Lower p-values indicate stronger evidence for a difference.")
    print("=" * 100)
    print(f"{'':<2}{'Pairwise Comparison':<60} {'P value':>12}")
    print("-" * 100)
    cur = None
    for section, label, p_str in rows:
        if section and section != cur:
            cur = section
            print(f"\n{section}")
        print(f"  {label:<60} {p_str:>12}")
    print()


def flatten_aligned_runs(a_mat: np.ndarray, b_mat: np.ndarray, max_runs: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Aligns (n_items, n_runs) matrices and flattens into 1D vectors of length n_items * n_runs_used.
    Uses up to max_runs runs (but never exceeding available runs in either matrix).
    """
    if a_mat is None or b_mat is None:
        return np.array([], dtype=int), np.array([], dtype=int), 0
    if a_mat.shape[0] != b_mat.shape[0]:
        # Align on min items if needed (defensive)
        n = min(a_mat.shape[0], b_mat.shape[0])
        a_mat = a_mat[:n, :]
        b_mat = b_mat[:n, :]
    runs_used = min(max_runs, a_mat.shape[1], b_mat.shape[1])
    if runs_used <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), 0
    a = a_mat[:, :runs_used].reshape(-1)
    b = b_mat[:, :runs_used].reshape(-1)
    return a, b, runs_used


def aggregate_runs_mean(a_mat: np.ndarray, b_mat: np.ndarray, max_runs: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Aligns (n_items, n_runs) matrices, takes per-item mean over runs (up to max_runs), and returns 1D vectors.
    This is the recommended way to avoid treating multiple runs of the SAME example as independent samples.
    """
    if a_mat is None or b_mat is None:
        return np.array([], dtype=float), np.array([], dtype=float), 0
    if a_mat.shape[0] != b_mat.shape[0]:
        n = min(a_mat.shape[0], b_mat.shape[0])
        a_mat = a_mat[:n, :]
        b_mat = b_mat[:n, :]
    runs_used = min(max_runs, a_mat.shape[1], b_mat.shape[1])
    if runs_used <= 0:
        return np.array([], dtype=float), np.array([], dtype=float), 0
    a = np.mean(a_mat[:, :runs_used], axis=1)
    b = np.mean(b_mat[:, :runs_used], axis=1)
    return a, b, runs_used


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_rows_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))


def fmt_p(p: float) -> str:
    if p != p:  # NaN
        return "nan"
    # Keep consistent with your prior CSV formatting in evaluation
    return f"{p:.10f}"


def fmt_p_display(p: float) -> str:
    """
    Format p-value for display purposes.
    Returns "<0.001" for p < 0.001, otherwise formats with appropriate precision.
    """
    if p != p:  # NaN
        return "nan"
    if p < 0.001:
        return "<0.001"
    # For p >= 0.001, show up to 3 decimal places
    if p < 0.01:
        return f"{p:.3f}"
    elif p < 0.1:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"


def print_model_table(model_name: str, table: List[Tuple[str, str, str, str]], show_value: bool = False) -> None:
    """
    table rows: (section_title_or_empty, comparison_label, value_str, p_str)
    """
    print("\n" + "=" * 100)
    print(f"Model: {model_name}")
    print("=" * 100)
    col1 = 62
    col2 = 55
    if show_value:
        col2 = 45
        print(f"{'Section / Comparison':<{col1}} {'Value':>10} {'P value':>10}")
    else:
        print(f"{'Section / Comparison':<{col1}} {'P value':>10}")
    print("-" * 100)
    current_section = None
    for section, comp_label, value_str, p_str in table:
        if section and section != current_section:
            current_section = section
            print(f"\n{section}")
        if show_value:
            print(f"  {comp_label:<{col2}} {value_str:>10} {p_str:>10}")
        else:
            print(f"  {comp_label:<{col2}} {p_str:>10}")
    print()


def write_vector_txt(path: Path, values: Sequence[int], as_percent: bool = False) -> None:
    """
    Write a numeric vector to a text file, one value per line.
    If as_percent=True, write 0/1 as 0.0000/100.0000 to resemble existing resA/resB files.
    """
    with path.open("w", encoding="utf-8") as f:
        for v in values:
            x = 100.0 * float(v) if as_percent else float(v)
            f.write(f"{x:.4f}\n")


def comparison_by_key(key: str) -> Comparison:
    for c in COMPARISONS:
        if c.key == key:
            return c
    raise KeyError(key)


def section_title(dataset_type: str, prompt_type: str) -> str:
    return SECTION_TITLES.get((dataset_type, prompt_type), f"{dataset_type} / {prompt_type}")


def write_pair_files(
    out_dir: Path,
    dataset_type: str,
    model_name: str,
    prompt_type: str,
    comparison_key: str,
    a: Sequence[Union[int, float]],
    b: Sequence[Union[int, float]],
    as_percent: bool = False,
) -> Tuple[Path, Path]:
    """
    Write files in a stat_prepare-like style so downstream code can refer to filename_A/filename_B.
    Returns (filename_A, filename_B).
    """
    # Mirror the style of `stat_prepare.py` where each model/dataset has its own folder.
    # Here we use: stat/{dataset_type}_{model_name}/{prompt_type}__{comparison_key}_A.txt and _B.txt
    model_folder = out_dir / f"{dataset_type}_{model_name}"
    model_folder.mkdir(parents=True, exist_ok=True)
    stem = f"{prompt_type}__{comparison_key}"
    filename_A = model_folder / f"{stem}_A.txt"
    filename_B = model_folder / f"{stem}_B.txt"
    write_vector_txt(filename_A, a, as_percent=as_percent)
    write_vector_txt(filename_B, b, as_percent=as_percent)
    return filename_A, filename_B


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired-permutation significance tables for Pokemon probing.")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--runs", type=int, default=3, help="Use up to this many runs per item (defaults to 7).")
    parser.add_argument("--mc", type=int, default=0, help="MC samples for non-binary metrics (0 disables).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run-aggregation",
        choices=["flatten", "mean", "counts"],
        default="counts",
        help=(
            "How to use multiple runs per example. "
            "'counts' (recommended for binary) sums runs per example (0..R) and uses exact integer DP (paper-consistent). "
            "'mean' averages runs per example then uses paired-permutation MC (25k) on floats. "
            "'flatten' treats each (example,run) as a separate paired observation (can inflate significance)."
        ),
    )
    parser.add_argument(
        "--write-resab",
        action="store_true",
        help=(
            "Write pooled vectors for the overall (across-model) test to stat/resA.txt and stat/resB.txt. "
            "By default this dumps the Generic+DrugIndication Default-vs-Mitigation comparison unless "
            "--resab-section/--resab-comparison override it."
        ),
    )
    parser.add_argument(
        "--resab-section",
        type=str,
        default="generic:drug_indication",
        help="Section to dump for resA/resB, format '<dataset_type>:<prompt_type>' e.g. 'brand:drug_dosing'.",
    )
    parser.add_argument(
        "--resab-comparison",
        type=str,
        default="default_vs_mitigation",
        choices=[c.key for c in COMPARISONS],
        help="Comparison key to dump for resA/resB (default: default_vs_mitigation).",
    )
    parser.add_argument(
        "--write-pair-files",
        action="store_true",
        help="Write per-model paired vectors (filename_A/filename_B) into evaluation/stat/ for all sections/comparisons.",
    )
    parser.add_argument(
        "--resab-percent",
        action="store_true",
        help="When writing resA/resB, output 0/1 values as 0.0000/100.0000 (to match the existing resA/resB style).",
    )
    parser.add_argument(
        "--table4",
        action="store_true",
        help=(
            "Print and save Table 4 pairwise model comparisons (as in the second figure). "
            "Uses per-example run-counts and exact paired-permutation DP (paper-consistent)."
        ),
    )
    parser.add_argument(
        "--table4-condition",
        choices=["default", "mitigation", "temp0"],
        default="default",
        help="Which condition to use when comparing models for Table 4 (default/no mitigation by default).",
    )
    parser.add_argument(
        "--table-generic-mitigation",
        action="store_true",
        help=(
            "Print and save pairwise model comparisons table for generic dataset with mitigation prompt. "
            "Shows comparisons for both Drug Dosing and Drug Indication prompts."
        ),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    all_rows: List[List[object]] = []

    # Collect pooled vectors (legacy; used for resA/resB and optional outputs):
    # pooled[(dataset_type, prompt_type, comparison_key)] = (listA, listB)
    pooled: Dict[Tuple[str, str, str], Tuple[List[float], List[float]]] = {}

    # Per-model printed tables (matches your screenshot layout; p-values differ per model).
    for dataset_type, model_name, model_dir in iter_model_dirs(results_dir):
        table_rows_for_print: List[Tuple[str, str, str, str]] = []

        for prompt_type in ("drug_indication", "drug_dosing"):
            section = SECTION_TITLES.get((dataset_type, prompt_type))
            if section is None:
                continue

            # Preload condition matrices
            mats = {
                cond: load_prompt_condition_matrix(model_dir, prompt_type, cond)
                for cond in ("default", "mitigation", "temp0")
            }

            # If none of the files exist for this prompt, skip section entirely.
            if all(m is None or m.size == 0 for m in mats.values()):
                continue

            for comp in COMPARISONS:
                a_mat = mats.get(comp.a_name)
                b_mat = mats.get(comp.b_name)
                if a_mat is None or b_mat is None or a_mat.size == 0 or b_mat.size == 0:
                    p = float("nan")
                    runs_used = 0
                    n_pairs = 0
                    mean_a = float("nan")
                    mean_b = float("nan")
                    delta = float("nan")
                else:
                    # Three ways to handle runs:
                    # - counts: per-example integer counts (0..R), exact DP p-value (paper-consistent).
                    # - mean: per-example mean (0..1), MC p-value on float deltas.
                    # - flatten: treat (example,run) as independent paired observations.
                    if args.run_aggregation == "counts":
                        u, v, runs_used = per_example_counts(a_mat, b_mat, max_runs=args.runs)
                        n_pairs = int(u.size)  # N examples
                        # Value: mean difference in suspicion rate
                        denom = float(n_pairs * runs_used) if (n_pairs and runs_used) else float("nan")
                        delta_total = float(np.sum(u - v)) if n_pairs else float("nan")
                        delta = float(delta_total / denom) if denom == denom and denom != 0.0 else float("nan")
                        mean_a = float(np.mean(u / runs_used)) if (n_pairs and runs_used) else float("nan")
                        mean_b = float(np.mean(v / runs_used)) if (n_pairs and runs_used) else float("nan")
                        p = exact_paired_perm_pvalue_integer_deltas((u - v).astype(int))
                        a_for_files: List[Union[int, float]] = u.tolist()
                        b_for_files: List[Union[int, float]] = v.tolist()
                    else:
                        if args.run_aggregation == "mean":
                            a, b, runs_used = aggregate_runs_mean(a_mat, b_mat, max_runs=args.runs)
                        else:
                            a, b, runs_used = flatten_aligned_runs(a_mat, b_mat, max_runs=args.runs)
                        n_pairs = int(a.size)
                        mean_a = float(np.mean(a)) if n_pairs else float("nan")
                        mean_b = float(np.mean(b)) if n_pairs else float("nan")
                        delta = float(mean_a - mean_b) if n_pairs else float("nan")
                        p = compute_pvalue(a, b, mc=args.mc, seed=args.seed)
                        a_for_files = [float(x) for x in np.asarray(a).ravel().tolist()]
                        b_for_files = [float(x) for x in np.asarray(b).ravel().tolist()]

                    # Add to pooled overall vectors
                    pool_key = (dataset_type, prompt_type, comp.key)
                    if pool_key not in pooled:
                        pooled[pool_key] = ([], [])
                    pooled[pool_key][0].extend([float(x) for x in a_for_files])
                    pooled[pool_key][1].extend([float(x) for x in b_for_files])

                    # Optionally write per-model A/B files like stat_prepare.py + testSig3.py expect
                    if args.write_pair_files:
                        write_pair_files(
                            out_dir=out_dir,
                            dataset_type=dataset_type,
                            model_name=model_name,
                            prompt_type=prompt_type,
                            comparison_key=comp.key,
                            a=a_for_files,
                            b=b_for_files,
                            as_percent=args.resab_percent,
                        )

                sig = "Yes" if (p == p and p <= args.alpha) else "No"
                p_str = fmt_p(p)
                value_str = f"{delta:+.4f}" if (delta == delta) else "nan"

                table_rows_for_print.append((section, comp.label, value_str, p_str))
                all_rows.append(
                    [
                        dataset_type,
                        model_name,
                        prompt_type,
                        comp.key,
                        comp.label,
                        value_str,
                        p_str,
                        sig,
                        n_pairs,
                        runs_used,
                        _safe_float(mean_a),
                        _safe_float(mean_b),
                        _safe_float(delta),
                    ]
                )

        # Print to main screen
        if table_rows_for_print:
            print_model_table(model_name=model_name, table=table_rows_for_print)

    # Print pooled overall table (across all models) and add rows to CSV
    # UPDATED overall (paper-consistent): per-example integer counts aggregated across all models×runs.
    #
    # For each example n:
    #   u_n = total suspicion_detected across all models×runs under condition A
    #   v_n = total suspicion_detected across all models×runs under condition B
    #
    # Then delta_n = u_n - v_n is an integer and the paired-permutation null is sign-flips per example.
    # We compute the exact p-value via DP over integer deltas and also write a paired-perm-test input file.
    overall_table_rows: List[Tuple[str, str, str, str]] = []
    pairedperm_dir = out_dir / "pairedperm_inputs"

    for dataset_type in ("generic", "brand"):
        for prompt_type in ("drug_indication", "drug_dosing"):
            section = SECTION_TITLES.get((dataset_type, prompt_type))
            if section is None:
                continue

            for comp in COMPARISONS:
                u_acc: Optional[np.ndarray] = None
                v_acc: Optional[np.ndarray] = None
                runs_per_model: List[int] = []
                models_used = 0

                for ds2, model_name, model_dir in iter_model_dirs(results_dir):
                    if ds2 != dataset_type:
                        continue

                    a_mat = load_prompt_condition_matrix(model_dir, prompt_type, comp.a_name)
                    b_mat = load_prompt_condition_matrix(model_dir, prompt_type, comp.b_name)
                    if a_mat is None or b_mat is None or a_mat.size == 0 or b_mat.size == 0:
                        continue

                    u_m, v_m, runs_used = per_example_counts(a_mat, b_mat, max_runs=args.runs)
                    if u_m.size == 0 or v_m.size == 0:
                        continue

                    if u_acc is None:
                        u_acc = u_m
                        v_acc = v_m
                    else:
                        n = min(u_acc.size, u_m.size)
                        u_acc = u_acc[:n] + u_m[:n]
                        v_acc = v_acc[:n] + v_m[:n]

                    runs_per_model.append(runs_used)
                    models_used += 1

                if u_acc is None or v_acc is None or u_acc.size == 0 or v_acc.size == 0:
                    p = float("nan")
                    n_pairs = 0
                    eff_runs = 0
                    mean_a = float("nan")
                    mean_b = float("nan")
                    delta = float("nan")
                else:
                    deltas = (u_acc - v_acc).astype(int)
                    p = exact_paired_perm_pvalue_integer_deltas(deltas)
                    n_pairs = int(u_acc.size)  # N examples
                    eff_runs = int(np.sum(runs_per_model))  # K = sum over models of runs_used (informational)
                    # Value: mean difference in suspicion rate across all models×runs
                    denom = float(n_pairs * eff_runs) if (n_pairs and eff_runs) else float("nan")
                    delta_total = float(np.sum(u_acc - v_acc)) if n_pairs else float("nan")
                    delta = float(delta_total / denom) if denom == denom and denom != 0.0 else float("nan")
                    mean_a = float(np.mean(u_acc / eff_runs)) if (n_pairs and eff_runs) else float("nan")
                    mean_b = float(np.mean(v_acc / eff_runs)) if (n_pairs and eff_runs) else float("nan")

                    # Write paired-perm-test input file (u_n,v_n per example)
                    out_path = pairedperm_dir / f"{dataset_type}__{prompt_type}__{comp.key}.csv"
                    write_pairedperm_input(out_path, u_acc.tolist(), v_acc.tolist())

                p_str = fmt_p(p)
                value_str = f"{delta:+.4f}" if (delta == delta) else "nan"
                overall_table_rows.append((section, comp.label, value_str, p_str))

                sig = "Yes" if (p == p and p <= args.alpha) else "No"
                all_rows.append(
                    [
                        f"{dataset_type}_overall_per_example_counts",
                        "__ALL_MODELS__",
                        prompt_type,
                        comp.key,
                        comp.label,
                        value_str,
                        p_str,
                        sig,
                        n_pairs,
                        eff_runs,
                        _safe_float(mean_a),
                        _safe_float(mean_b),
                        _safe_float(delta),
                    ]
                )

    if overall_table_rows:
        print_model_table(
            model_name="__ALL_MODELS__ (per-example counts across all models×runs)",
            table=overall_table_rows,
        )
        print(f"Paired-perm-test inputs written under: {pairedperm_dir}")

    # Optionally dump pooled vectors to resA/resB
    if args.write_resab:
        try:
            ds_type, pr_type = args.resab_section.split(":", 1)
        except Exception:
            raise ValueError("--resab-section must be '<dataset_type>:<prompt_type>' e.g. 'generic:drug_indication'")
        pool_key = (ds_type, pr_type, args.resab_comparison)
        a_list, b_list = pooled.get(pool_key, ([], []))
        if not a_list or not b_list or len(a_list) != len(b_list):
            raise RuntimeError(f"No pooled data available for {pool_key}; cannot write resA/resB.")
        resA_path = out_dir / "resA.txt"
        resB_path = out_dir / "resB.txt"
        write_vector_txt(resA_path, a_list, as_percent=args.resab_percent)
        write_vector_txt(resB_path, b_list, as_percent=args.resab_percent)
        print(
            f"Wrote pooled vectors: {resA_path} and {resB_path} "
            f"(section={args.resab_section}, comparison={args.resab_comparison})"
        )

        # Also write the same pooled vectors into a stable "overall" subfolder, one file per comparison,
        # so you can point `testSig3.py`/paired-perm-test to a deterministic filename_A/filename_B.
        overall_folder = out_dir / "overall"
        overall_folder.mkdir(parents=True, exist_ok=True)
        stem = f"{args.resab_section.replace(':', '__')}__{args.resab_comparison}"
        overall_A = overall_folder / f"{stem}_A.txt"
        overall_B = overall_folder / f"{stem}_B.txt"
        write_vector_txt(overall_A, a_list, as_percent=args.resab_percent)
        write_vector_txt(overall_B, b_list, as_percent=args.resab_percent)
        print(f"Wrote overall pair files: {overall_A} and {overall_B}")

    # Table 4: Pairwise model comparisons within each dataset_type × prompt_type
    if args.table4:
        table4_rows: List[Tuple[str, str, str]] = []
        table4_csv_rows: List[List[object]] = []
        # Determine which base models exist (normalize gpt variants)
        # We'll use internal keys expected in RESULTS_DIR: llama, gemma, qwen, gpt4o/gpt_4o
        preferred_models = ["llama", "gemma", "gpt4o", "gpt_4o", "qwen"]

        for dataset_type in ("generic", "brand"):
            for prompt_type in ("drug_dosing", "drug_indication"):
                section = (
                    ("Drug Dosing Prompt (generic drug dataset)" if dataset_type == "generic" else "Drug Dosing Prompt (brand drug dataset)")
                    if prompt_type == "drug_dosing"
                    else ("Drug Indication Prompt (generic drug dataset)" if dataset_type == "generic" else "Drug Indication Prompt (brand drug dataset)")
                )

                # Load one matrix per model for the selected condition
                mats_by_model: Dict[str, np.ndarray] = {}
                for ds2, model_name, model_dir in iter_model_dirs(results_dir):
                    if ds2 != dataset_type:
                        continue
                    if model_name not in preferred_models:
                        continue
                    mat = load_prompt_condition_matrix(model_dir, prompt_type, args.table4_condition)
                    if mat is None or mat.size == 0:
                        continue
                    mats_by_model[model_name] = mat

                # Choose a stable list of models actually available in this section
                models_avail = [m for m in preferred_models if m in mats_by_model]
                # Prefer gpt4o naming; if both exist, keep only one to avoid duplicates
                if "gpt4o" in models_avail and "gpt_4o" in models_avail:
                    models_avail = [m for m in models_avail if m != "gpt_4o"]

                for a_model, b_model in _model_pairs(models_avail):
                    mat_a = mats_by_model[a_model]
                    mat_b = mats_by_model[b_model]
                    value, p, n, r = pairwise_model_pvalue_counts(mat_a, mat_b, max_runs=args.runs)
                    p_str = fmt_p(p)
                    a_name = MODEL_DISPLAY.get(a_model, a_model)
                    b_name = MODEL_DISPLAY.get(b_model, b_model)
                    label = f"{a_name} vs. {b_name}"
                    table4_rows.append((section, label, p_str))
                    table4_csv_rows.append(
                        [
                            dataset_type,
                            prompt_type,
                            args.table4_condition,
                            a_model,
                            b_model,
                            label,
                            f"{value:+.6f}" if value == value else "nan",
                            p_str,
                            n,
                            r,
                        ]
                    )

        print_pairwise_table4(table4_rows)
        out_table4 = out_dir / f"Table4_pairwise_model_comparisons_{args.table4_condition}.csv"
        write_rows_csv(
            out_table4,
            [
                "dataset_type",
                "prompt_type",
                "condition",
                "model_A",
                "model_B",
                "label",
                "value(mean_rate_A_minus_B)",
                "p_value",
                "n_examples",
                "runs_used",
            ],
            table4_csv_rows,
        )
        print(f"Saved Table 4 CSV: {out_table4}")

    # Table: Generic Dataset + Mitigation Prompt pairwise comparisons
    if args.table_generic_mitigation:
        generic_mitigation_rows: List[Tuple[str, str, str]] = []
        generic_mitigation_csv_rows: List[List[object]] = []
        # Order: llama, gpt4o, qwen, gemma
        # Note: Directory names might be "llama3" instead of "llama", so we need to handle both
        # The _model_pairs function will generate pairs in order:
        # llama vs gpt4o, llama vs qwen, llama vs gemma, gpt4o vs qwen, gpt4o vs gemma, qwen vs gemma
        # But user wants: llama vs all, then gemma vs remaining, then gpt4o vs qwen
        # So we use: llama, gemma, gpt4o, qwen which gives:
        # llama vs gemma, llama vs gpt4o, llama vs qwen, gemma vs gpt4o, gemma vs qwen, gpt4o vs qwen
        # Then we reorder to match user's format
        preferred_models = ["llama", "llama3", "gemma", "gpt4o", "gpt_4o", "qwen"]
        dataset_type = "generic"
        condition = "mitigation"

        for prompt_type in ("drug_dosing", "drug_indication"):
            section = get_generic_mitigation_section_title(prompt_type)

            # Load one matrix per model for the mitigation condition
            mats_by_model: Dict[str, np.ndarray] = {}
            for ds2, model_name, model_dir in iter_model_dirs(results_dir):
                if ds2 != dataset_type:
                    continue
                if model_name not in preferred_models:
                    continue
                mat = load_prompt_condition_matrix(model_dir, prompt_type, condition)
                if mat is None or mat.size == 0:
                    continue
                mats_by_model[model_name] = mat
            
            # Debug: Print which models were found
            if not mats_by_model:
                print(f"Warning: No models found for {prompt_type} with {condition} condition")
                continue
            
            # Debug output: show which models were successfully loaded
            print(f"Loaded {len(mats_by_model)} models for {prompt_type}: {list(mats_by_model.keys())}")

            # Normalize model names: handle gpt4o/gpt_4o and llama/llama3 variations
            # Map directory names to canonical names for consistent pairing
            model_name_map = {}
            for model_key in mats_by_model.keys():
                if model_key == "gpt_4o" or model_key == "gpt4o":
                    model_name_map["gpt4o"] = model_key  # Use first occurrence
                elif model_key == "llama3" or model_key == "llama":
                    model_name_map["llama"] = model_key  # Normalize llama3 -> llama
                else:
                    model_name_map[model_key] = model_key
            
            # Get available models in preferred order, using canonical names
            models_avail_canonical = []
            canonical_preferred = ["llama", "gemma", "gpt4o", "qwen"]  # Canonical order
            for preferred in canonical_preferred:
                if preferred in model_name_map:
                    models_avail_canonical.append(preferred)
                elif preferred == "gpt4o" and ("gpt4o" in model_name_map or "gpt_4o" in mats_by_model):
                    models_avail_canonical.append("gpt4o")
                elif preferred == "llama" and ("llama" in model_name_map or "llama3" in mats_by_model):
                    models_avail_canonical.append("llama")
            
            # Remove duplicates (in case both gpt4o and gpt_4o were added)
            models_avail_canonical = list(dict.fromkeys(models_avail_canonical))
            
            # Generate all possible pairs first
            pairs = _model_pairs(models_avail_canonical)
            
            # Reorder pairs to match exact user's desired format:
            # 1. Llama-3.3-70B-Instruct vs. GPT-4o-mini
            # 2. Llama-3.3-70B-Instruct vs. Qwen3
            # 3. Llama-3.3-70B-Instruct vs. Gemma-3-27B-IT
            # 4. Gemma-3-27B-IT vs. GPT-4o-mini
            # 5. Gemma-3-27B-IT vs. Qwen3
            # 6. GPT-4o-mini vs. Qwen3
            def find_pair(pairs_list, a, b):
                """Find pair (a,b) or (b,a) in pairs_list"""
                for p in pairs_list:
                    if (p[0] == a and p[1] == b) or (p[0] == b and p[1] == a):
                        return p
                return None
            
            ordered_pairs = []
            desired_order = [
                ("llama", "gpt4o"), ("llama", "qwen"), ("llama", "gemma"),
                ("gemma", "gpt4o"), ("gemma", "qwen"), ("gpt4o", "qwen")
            ]
            
            # Build ordered pairs list in exact desired order
            # Always attempt all 6 desired pairs, even if models are missing
            for a, b in desired_order:
                if a in models_avail_canonical and b in models_avail_canonical:
                    pair = find_pair(pairs, a, b)
                    if pair:
                        ordered_pairs.append(pair)
                else:
                    # Report missing models for this pair
                    missing = []
                    if a not in models_avail_canonical:
                        missing.append(a)
                    if b not in models_avail_canonical:
                        missing.append(b)
                    if missing:
                        print(f"Warning: Missing models for pair {a} vs {b}: {missing} (available: {models_avail_canonical})")
            
            # Add any remaining pairs that weren't in desired order (shouldn't happen if all 6 are present)
            for pair in pairs:
                if pair not in ordered_pairs:
                    ordered_pairs.append(pair)
            
            # Ensure we have exactly 6 pairs if all models are available
            if len(models_avail_canonical) == 4 and len(ordered_pairs) != 6:
                print(f"Warning: Expected 6 pairs but got {len(ordered_pairs)} for {prompt_type}. Available models: {models_avail_canonical}, Pairs: {ordered_pairs}")
            elif len(models_avail_canonical) < 4:
                print(f"Warning: Only {len(models_avail_canonical)} models available for {prompt_type} (expected 4). Available: {models_avail_canonical}")
            
            # Map canonical names back to actual directory names for matrix lookup
            def get_actual_model_name(canonical_name):
                """Get the actual model name from directory"""
                if canonical_name == "gpt4o":
                    # Check which variant exists
                    if "gpt4o" in mats_by_model:
                        return "gpt4o"
                    elif "gpt_4o" in mats_by_model:
                        return "gpt_4o"
                elif canonical_name == "llama":
                    # Check which variant exists
                    if "llama3" in mats_by_model:
                        return "llama3"
                    elif "llama" in mats_by_model:
                        return "llama"
                return canonical_name

            for a_model_canonical, b_model_canonical in ordered_pairs:
                # Get actual model names from directory
                a_model = get_actual_model_name(a_model_canonical)
                b_model = get_actual_model_name(b_model_canonical)
                
                if a_model not in mats_by_model or b_model not in mats_by_model:
                    continue
                    
                mat_a = mats_by_model[a_model]
                mat_b = mats_by_model[b_model]
                value, p, n, r = pairwise_model_pvalue_counts(mat_a, mat_b, max_runs=args.runs)
                p_str_display = fmt_p_display(p)  # For display
                p_str_csv = fmt_p(p)  # For CSV (precise)
                # Use canonical names for display to ensure consistent naming
                a_name = MODEL_DISPLAY.get(a_model_canonical, a_model_canonical)
                b_name = MODEL_DISPLAY.get(b_model_canonical, b_model_canonical)
                label = f"{a_name} vs. {b_name}"
                generic_mitigation_rows.append((section, label, p_str_display))
                generic_mitigation_csv_rows.append(
                    [
                        dataset_type,
                        prompt_type,
                        condition,
                        a_model_canonical,  # Use canonical name for CSV
                        b_model_canonical,  # Use canonical name for CSV
                        label,
                        f"{value:+.6f}" if value == value else "nan",
                        p_str_csv,  # Use precise value for CSV
                        n,
                        r,
                    ]
                )

        print_generic_mitigation_table(generic_mitigation_rows)
        out_generic_mitigation = out_dir / "Table_generic_mitigation_pairwise_comparisons.csv"
        write_rows_csv(
            out_generic_mitigation,
            [
                "dataset_type",
                "prompt_type",
                "condition",
                "model_A",
                "model_B",
                "label",
                "value(mean_rate_A_minus_B)",
                "p_value",
                "n_examples",
                "runs_used",
            ],
            generic_mitigation_csv_rows,
        )
        print(f"Saved Generic Mitigation Table CSV: {out_generic_mitigation}")

    # Save combined CSV
    out_csv = out_dir / "Permutation_pair_statistical_test_results_pokemon.csv"
    header = [
        "dataset_type",
        "model",
        "prompt_type",
        "comparison_key",
        "comparison_label",
        "value(mean_rate_A_minus_B)",
        "p_value",
        "significant(alpha)",
        "n_pairs(items*runs)",
        "runs_used",
        "mean_A(hallucinations)",
        "mean_B(hallucinations)",
        "delta(A-B)",
    ]
    write_rows_csv(out_csv, header, all_rows)
    print(f"Saved CSV: {out_csv}")


if __name__ == "__main__":
    main()


