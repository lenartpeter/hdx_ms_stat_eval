"""
Copyright 2026 Péter Lénárt

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""
================================================================================
HDX-MS EQUIVALENCE TESTING - STATISTICS MODULE
================================================================================

PURPOSE:
    Shared statistical calculations used across multiple analysis modules.

INPUTS:
    - Uptake value arrays (numpy arrays or lists)
    - Statistical parameters (alpha, degrees of freedom, sample sizes)

OUTPUTS:
    - Calculated statistics (SE, df, CI bounds, delta_D)
    - Derived values (Sidak-corrected alpha, number of pairings)

DEPENDENCIES:
    - numpy
    - scipy.stats

USAGE:
    from statistics import (
        calculate_welch_ci,
        calculate_sidak_alpha,
        calculate_n_pairings
    )
    
    result = calculate_welch_ci(uptakes_ref, uptakes_sample, alpha_ci=0.10)
    alpha_s = calculate_sidak_alpha(0.05, m=2)
    n_pairings = calculate_n_pairings(n_replicates=8, group_size=3)

================================================================================
"""

import numpy as np
from scipy import stats
from math import comb
from typing import Dict, Tuple


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Number of hypothesis tests (Šidák correction parameter)
SIDAK_M_PARAMETER = 2


# ==============================================================================
# T-VALUE CACHE
# ==============================================================================

# Module-level cache for t-critical values to avoid repeated scipy calls
# Key: (rounded_df, alpha), Value: t_critical
_T_VALUE_CACHE: Dict[Tuple[float, float], float] = {}


def get_cached_t_value(df: float, alpha: float = 0.10) -> float:
    """
    Get t-critical value with caching for performance.
    
    Since many calculations have similar degrees of freedom, caching
    significantly speeds up repeated computations (e.g., during
    precomputation of 314k+ values).
    
    Parameters
    ----------
    df : float
        Degrees of freedom (can be non-integer from Satterthwaite)
    alpha : float, optional
        Significance level for two-sided test (default 0.10)
        
    Returns
    -------
    float
        Critical t-value for the given df and alpha
        Returns NaN if df is invalid (NaN or <= 0)
        
    Notes
    -----
    df is rounded to 4 decimal places for cache key to balance
    accuracy with cache efficiency.
    """
    if np.isnan(df) or df <= 0:
        return np.nan
    
    # Round only the cache key to reduce cache size without changing
    # the value used in the calculation.
    df_rounded = round(df, 4)
    cache_key = (df_rounded, alpha)
    
    if cache_key not in _T_VALUE_CACHE:
        _T_VALUE_CACHE[cache_key] = stats.t.ppf(1 - alpha / 2, df)
    
    return _T_VALUE_CACHE[cache_key]


def clear_t_cache() -> None:

    global _T_VALUE_CACHE
    _T_VALUE_CACHE = {}


# ==============================================================================
# CORE STATISTICAL FUNCTIONS
# ==============================================================================

def calculate_satterthwaite_df(sd_ref: float, sd_sample: float,
                                n_ref: int, n_sample: int) -> float:
    """
    Calculate degrees of freedom using Satterthwaite [Satterthwaite, F. E. An Approximate Distribution of
    Estimates of Variance Components. Biom. Bull. 1946, 2 (6), 110–114. https://doi.org/10.2307/3002019.] approximation.
    
    Formula:
        df = (s1²/n1 + s2²/n2)² / ((s1²/n1)²/(n1-1) + (s2²/n2)²/(n2-1))
    
    Parameters
    ----------
    sd_ref : float
        Standard deviation of reference group
    sd_sample : float
        Standard deviation of sample group
    n_ref : int
        Sample size of reference group
    n_sample : int
        Sample size of sample group
        
    Returns
    -------
    float
        Degrees of freedom (may be non-integer)
        
    Notes
    -----
    If both standard deviations are zero (all values identical within groups),
    returns n_ref + n_sample - 2 as a fallback.

    """
    # Handle edge case where both SDs are zero
    if sd_ref == 0 and sd_sample == 0:
        return n_ref + n_sample - 2
    
    var_ref = sd_ref ** 2
    var_sample = sd_sample ** 2
    
    numerator = (var_ref / n_ref + var_sample / n_sample) ** 2
    
    denominator = ((var_ref / n_ref) ** 2 / (n_ref - 1) +
                   (var_sample / n_sample) ** 2 / (n_sample - 1))
    
    if denominator == 0:
        return n_ref + n_sample - 2
    
    return numerator / denominator


def calculate_standard_error(sd_ref: float, sd_sample: float,
                              n_ref: int, n_sample: int) -> float:
    """
    Calculate standard error of the difference between two means.
    
    Formula:
        SE = sqrt(s1²/n1 + s2²/n2)
    
    Parameters
    ----------
    sd_ref : float
        Standard deviation of reference group
    sd_sample : float
        Standard deviation of sample group
    n_ref : int
        Sample size of reference group
    n_sample : int
        Sample size of sample group
        
    Returns
    -------
    float
        Standard error of the difference
    """
    return np.sqrt(sd_ref ** 2 / n_ref + sd_sample ** 2 / n_sample)


def calculate_welch_ci(uptakes_ref: np.ndarray, uptakes_sample: np.ndarray,
                        alpha_ci: float = 0.10) -> Dict[str, float]:
    """
    Calculate delta_D and confidence interval using Welch's t-test.
    
    This function computes all statistics needed for equivalence testing:
    - delta_D (difference in means)
    - Confidence interval bounds
    - Standard error
    - Degrees of freedom
    
    Parameters
    ----------
    uptakes_ref : np.ndarray
        Deuterium uptake values for reference group
    uptakes_sample : np.ndarray
        Deuterium uptake values for sample group
    alpha_ci : float, optional
        Significance level for CI (default 0.10 for 90% CI)
        
    Returns
    -------
    dict
        Dictionary containing:
        - delta_d: Difference in means (sample - reference)
        - ci_lp: Lower bound of confidence interval
        - ci_up: Upper bound of confidence interval
        - ci_lp_up_abs_max: max(|ci_lp|, |ci_up|)
        - se: Standard error
        - sd_ref: Standard deviation of reference group
        - sd_sample: Standard deviation of sample group
        - df: Degrees of freedom
        - t_critical: Critical t-value used
        - mean_ref: Mean of reference group
        - mean_sample: Mean of sample group
        
    Notes
    -----
    Uses Satterthwaite approximation for degrees of freedom and
    cached t-values for performance.
    """
    uptakes_ref = np.asarray(uptakes_ref)
    uptakes_sample = np.asarray(uptakes_sample)
    
    n_ref = len(uptakes_ref)
    n_sample = len(uptakes_sample)
    
    mean_ref = np.mean(uptakes_ref)
    mean_sample = np.mean(uptakes_sample)
    
    # Sample std with Bessel's correction (ddof=1)
    sd_ref = np.std(uptakes_ref, ddof=1)
    sd_sample = np.std(uptakes_sample, ddof=1)
    
    # Difference in means (sample - reference)
    delta_d = mean_sample - mean_ref
    
    # Satterthwaite degrees of freedom
    df = calculate_satterthwaite_df(sd_ref, sd_sample, n_ref, n_sample)
    
    # Standard error
    se = calculate_standard_error(sd_ref, sd_sample, n_ref, n_sample)
    
    # Critical t-value for two-sided CI (cached)
    t_critical = get_cached_t_value(df, alpha_ci)
    
    # Margin of error
    margin = t_critical * se
    
    # Confidence interval bounds
    ci_lp = delta_d - margin
    ci_up = delta_d + margin
    
    # Maximum absolute CI bound (used for EAC)
    ci_lp_up_abs_max = max(abs(ci_lp), abs(ci_up))
    
    return {
        'delta_d': delta_d,
        'ci_lp': ci_lp,
        'ci_up': ci_up,
        'ci_lp_up_abs_max': ci_lp_up_abs_max,
        'se': se,
        'sd_ref': sd_ref,
        'sd_sample': sd_sample,
        'df': df,
        't_critical': t_critical,
        'mean_ref': mean_ref,
        'mean_sample': mean_sample
    }


# ==============================================================================
# DERIVED STATISTICAL VALUES
# ==============================================================================

def calculate_sidak_alpha(alpha: float, m: int = SIDAK_M_PARAMETER) -> float:
    """
    Calculate Šidák-corrected [Šidák, Z. Rectangular Confidence Regions for the Means of Multivariate Normal
    Distributions. J. Am. Stat. Assoc. 1967, 62 (318), 626–633. https://doi.org/10.1080/01621459.1967.10482935]
    significance level for multiple testing.
    
    Formula:
        α_s = 1 - (1 - α)^(1/m)
    
    Parameters
    ----------
    alpha : float
        Family-wise significance level (e.g., 0.05)
    m : int, optional
        Number of hypothesis tests (default SIDAK_M_PARAMETER = 2)
        
    Returns
    -------
    float
        Šidák-corrected significance level

    """
    return 1 - (1 - alpha) ** (1 / m)


def calculate_percentile_from_alpha(alpha_s: float) -> float:
    """
    Convert Šidák-corrected alpha to the corresponding percentile.
    
    Parameters
    ----------
    alpha_s : float
        Šidák-corrected significance level
        
    Returns
    -------
    float
        Percentile value (e.g., 97.47 for α_s ≈ 0.0253)
    """
    return 100 * (1 - alpha_s)


def calculate_n_pairings(n_replicates: int, group_size: int) -> int:
    """
    Calculate the number of non-overlapping replicate pairings.
    
    For splitting n replicates into two non-overlapping groups of
    size k (group_size), the number of unique pairings is:
    
        C(n, k) × C(n-k, k) / 2
    
    The division by 2 avoids counting pairs where reference and sample
    assignments are swapped (which would yield identical ΔD magnitudes).
    
    Parameters
    ----------
    n_replicates : int
        Total number of replicates (e.g., 8)
    group_size : int
        Size of each group (e.g., 3)
        
    Returns
    -------
    int
        Number of unique non-overlapping pairings
        
    Raises
    ------
    ValueError
        If n_replicates < 2 * group_size (cannot form two non-overlapping groups)
        
    Examples
    --------
    >>> calculate_n_pairings(8, 3)
    280
    
    >>> calculate_n_pairings(10, 3)
    840
    
    Notes
    -----
    For n=8, k=3: C(8,3) × C(5,3) / 2 = 56 × 10 / 2 = 280
    """
    if n_replicates < 2 * group_size:
        raise ValueError(
            f"n_replicates ({n_replicates}) must be at least "
            f"2 × group_size ({2 * group_size}) for non-overlapping pairings"
        )
    
    # C(n, k) - ways to choose first group
    first_group = comb(n_replicates, group_size)
    
    # C(n-k, k) - ways to choose second group from remaining
    second_group = comb(n_replicates - group_size, group_size)
    
    # Divide by 2 to avoid counting (A,B) and (B,A) as different
    return (first_group * second_group) // 2


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    print("STATISTICS MODULE - TEST")
    print("-" * 60)
    
    # Test data - two groups of uptake values
    ref = np.array([1.0, 1.1, 0.9])
    sample = np.array([1.5, 1.6, 1.4])
    
    print(f"\nTest data:")
    print(f"  Reference: {ref}")
    print(f"  Sample: {sample}")
    
    # Test calculate_welch_ci
    result = calculate_welch_ci(ref, sample, alpha_ci=0.10)
    
    print(f"\nWelch CI results:")
    print(f"  delta_D = {result['delta_d']:.4f}")
    print(f"  CI = [{result['ci_lp']:.4f}, {result['ci_up']:.4f}]")
    print(f"  |CI|_max = {result['ci_lp_up_abs_max']:.4f}")
    print(f"  SE = {result['se']:.4f}")
    print(f"  df = {result['df']:.2f}")
    print(f"  t_critical = {result['t_critical']:.4f}")
    
    # Test Sidak correction
    print(f"\nŠidák correction (α=0.05, m=2):")
    alpha_s = calculate_sidak_alpha(0.05, m=2)
    percentile = calculate_percentile_from_alpha(alpha_s)
    print(f"  α_s = {alpha_s:.6f}")
    print(f"  Percentile = {percentile:.4f}th")
    
    # Test n_pairings calculation
    print(f"\nNumber of pairings:")
    for n_rep, g_size in [(8, 3), (10, 3), (8, 4), (12, 4)]:
        n_pair = calculate_n_pairings(n_rep, g_size)
        print(f"  n_replicates={n_rep}, group_size={g_size} → {n_pair} pairings")
    
    # Test t-value cache
    print(f"\nT-value cache test:")
    _ = get_cached_t_value(4.0, 0.10)
    _ = get_cached_t_value(4.0, 0.10)  # Should hit cache
    _ = get_cached_t_value(5.5, 0.10)
    print(f"  Cache size after 3 calls (2 unique): {len(_T_VALUE_CACHE)}")
    
    clear_t_cache()
    print(f"  Cache size after clear: {len(_T_VALUE_CACHE)}")
    
    print("\n" + "-" * 60)
    print("TEST COMPLETE")
