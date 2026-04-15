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
HDX-MS EQUIVALENCE TESTING - OUTLIER DETECTION MODULE
================================================================================

PURPOSE:
    Pure outlier detection algorithms operating on numerical arrays.
    No HDX-MS specific logic - can be tested and discussed in isolation.

DEPENDENCIES:
    - numpy
    - scipy.stats
    - statsmodels.stats.stattools (REQUIRED for adjusted_boxplot)

EXPORTED:
    - detect_outliers(values, method, cutoff, verbose) -> result_dict
    - Individual *_core() functions for each method
    - AVAILABLE_METHODS: list of valid method names
    - DEFAULT_CUTOFFS: dict of default cutoff values per method

METHODS AVAILABLE:
    - boxplot (default cutoff=1.5)
    - adjusted_boxplot (default cutoff=1.5, uses medcouple)
    - hampel (default cutoff=3.5)

USAGE:
    from outlier_detection import detect_outliers

    values = [1.0, 1.1, 1.2, 1.1, 5.0, 1.0, 1.2, 1.1]  # 5.0 is outlier
    result = detect_outliers(values, method='boxplot', cutoff=1.5)

    if result is not None:
        print(f"Outliers at indices: {result['outlier_indices']}")
        print(f"Number of outliers: {result['n_outliers']}")

================================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union


# ==============================================================================
# CONSTANTS
# ==============================================================================

DEFAULT_CUTOFFS = {
    'boxplot': 1.5,
    'adjusted_boxplot': 1.5,
    'hampel': 3.5,
}

AVAILABLE_METHODS = list(DEFAULT_CUTOFFS.keys())

# Minimum number of values required for reliable outlier detection
MIN_VALUES_REQUIRED = 4


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _validate_and_clean_input(values: Union[np.ndarray, List[float]],
                               min_size: int = MIN_VALUES_REQUIRED) -> Optional[np.ndarray]:
    """
    Validate and clean input array.

    Parameters
    ----------
    values : array-like
        Input values to validate
    min_size : int
        Minimum number of non-NaN values required

    Returns
    -------
    np.ndarray or None
        Cleaned array with NaN removed, or None if insufficient data
    """
    values = np.asarray(values, dtype=float)

    # Remove NaN values
    values = values[~np.isnan(values)]

    if len(values) < min_size:
        return None

    return values


def _calculate_medcouple(values: np.ndarray) -> float:
    """
    Calculate medcouple statistic for skewness measurement.

    Medcouple is a robust measure of skewness that ranges from -1 to 1:
    - MC > 0: right-skewed
    - MC < 0: left-skewed
    - MC = 0: symmetric

    Parameters
    ----------
    values : np.ndarray
        Input values (should not contain NaN)

    Returns
    -------
    float
        Medcouple value

    Raises
    ------
    ImportError
        If statsmodels is not available
    """
    try:
        from statsmodels.stats.stattools import medcouple
    except ImportError:
        raise ImportError(
            "statsmodels is required for adjusted_boxplot method. "
            "Install it with: pip install statsmodels"
        )

    return medcouple(values)


def _build_result_dict(outlier_indices: List[int],
                       n_total: int,
                       method: str,
                       cutoff: float,
                       **kwargs) -> Dict[str, Any]:
    """
    Build standardized result dictionary.

    Parameters
    ----------
    outlier_indices : list
        Indices of detected outliers
    n_total : int
        Total number of values analyzed
    method : str
        Method name
    cutoff : float
        Cutoff value used
    **kwargs : dict
        Additional method-specific fields

    Returns
    -------
    dict
        Standardized result dictionary
    """
    n_outliers = len(outlier_indices)
    outlier_percentage = (n_outliers / n_total * 100) if n_total > 0 else 0.0

    result = {
        'outlier_indices': outlier_indices,
        'n_outliers': n_outliers,
        'n_total': n_total,
        'outlier_percentage': outlier_percentage,
        'method': method,
        'cutoff': cutoff,
    }

    # Add method-specific fields
    result.update(kwargs)

    return result


def _print_detection_summary(result: Dict[str, Any], verbose: bool) -> None:
    """
    Print detection summary if verbose mode is enabled.
    """
    if not verbose:
        return

    method = result['method']
    cutoff = result['cutoff']
    n_total = result['n_total']
    n_outliers = result['n_outliers']
    pct = result['outlier_percentage']

    print(f"\n{'='*70}")
    print(f"{method.upper()} OUTLIER DETECTION (cutoff={cutoff})")
    print(f"{'='*70}")
    print(f"Total values: {n_total}")
    print(f"Outliers detected: {n_outliers} ({pct:.1f}%)")

    if 'median' in result:
        print(f"Median: {result['median']:.6f}")

    if 'q1' in result and 'q3' in result:
        print(f"Q1: {result['q1']:.6f}  |  Q3: {result['q3']:.6f}  |  IQR: {result['iqr']:.6f}")

    if 'mad' in result:
        print(f"MAD: {result['mad']:.6f}")

    if 'medcouple' in result:
        mc = result['medcouple']
        skew_type = "right-skewed" if mc > 0 else "left-skewed" if mc < 0 else "symmetric"
        print(f"Medcouple: {mc:.4f} ({skew_type})")

    if 'lower_bound' in result and 'upper_bound' in result:
        print(f"Bounds: [{result['lower_bound']:.6f}, {result['upper_bound']:.6f}]")

    print(f"{'='*70}")


# ==============================================================================
# CORE DETECTION FUNCTIONS
# ==============================================================================

def boxplot_outlier_detection_core(values: Union[np.ndarray, List[float]],
                                    cutoff: float = 1.5,
                                    verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    Standard IQR boxplot method for outlier detection.

    Outliers are values outside [Q1 - k*IQR, Q3 + k*IQR] where k is the cutoff.

    Parameters
    ----------
    values : array-like
        Input values (NaN values are removed internally)
    cutoff : float
        Multiplier for IQR (default 1.5, use 3.0 for extreme outliers only)
    verbose : bool
        Print detection statistics

    Returns
    -------
    dict or None
        Result dictionary with outlier_indices and statistics,
        or None if insufficient data (<4 values)
    """
    values = _validate_and_clean_input(values)
    if values is None:
        return None

    # Calculate quartiles and IQR
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    median = float(np.median(values))

    # Handle zero IQR (all values in middle 50% are identical)
    if iqr == 0:
        result = _build_result_dict(
            outlier_indices=[],
            n_total=len(values),
            method='boxplot',
            cutoff=cutoff,
            q1=q1, q3=q3, iqr=iqr, median=median,
            lower_bound=q1, upper_bound=q3
        )
        _print_detection_summary(result, verbose)
        return result

    # Calculate bounds
    lower_bound = q1 - cutoff * iqr
    upper_bound = q3 + cutoff * iqr

    # Identify outliers
    outlier_indices = []
    for idx, val in enumerate(values):
        if val < lower_bound or val > upper_bound:
            outlier_indices.append(idx)

    result = _build_result_dict(
        outlier_indices=outlier_indices,
        n_total=len(values),
        method='boxplot',
        cutoff=cutoff,
        q1=q1, q3=q3, iqr=iqr, median=median,
        lower_bound=lower_bound, upper_bound=upper_bound
    )

    _print_detection_summary(result, verbose)
    return result


def adjusted_boxplot_outlier_detection_core(values: Union[np.ndarray, List[float]],
                                             cutoff: float = 1.5,
                                             verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    Adjusted boxplot method using medcouple for skewness correction.

    This method adjusts the whisker lengths based on the skewness of the data.

    For MC >= 0 (right-skewed or symmetric):
        lower = Q1 - k * exp(-4*MC) * IQR
        upper = Q3 + k * exp(3*MC) * IQR

    For MC < 0 (left-skewed):
        lower = Q1 - k * exp(-3*MC) * IQR
        upper = Q3 + k * exp(4*MC) * IQR

    Parameters
    ----------
    values : array-like
        Input values (NaN values are removed internally)
    cutoff : float
        Base multiplier for whisker length (default 1.5)
    verbose : bool
        Print detection statistics

    Returns
    -------
    dict or None
        Result dictionary with outlier_indices and statistics,
        or None if insufficient data

    Raises
    ------
    ImportError
        If statsmodels is not available
    """
    values = _validate_and_clean_input(values)
    if values is None:
        return None

    # Calculate quartiles and IQR
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    median = float(np.median(values))

    # Handle zero IQR
    if iqr == 0:
        result = _build_result_dict(
            outlier_indices=[],
            n_total=len(values),
            method='adjusted_boxplot',
            cutoff=cutoff,
            q1=q1, q3=q3, iqr=iqr, median=median,
            medcouple=0.0,
            lower_bound=q1, upper_bound=q3
        )
        _print_detection_summary(result, verbose)
        return result

    # Calculate medcouple (will raise ImportError if statsmodels not available)
    mc = _calculate_medcouple(values)

    # Calculate adjusted bounds based on skewness
    if mc >= 0:  # Right-skewed or symmetric
        lower_bound = q1 - cutoff * np.exp(-4 * mc) * iqr
        upper_bound = q3 + cutoff * np.exp(3 * mc) * iqr
    else:  # Left-skewed
        lower_bound = q1 - cutoff * np.exp(-3 * mc) * iqr
        upper_bound = q3 + cutoff * np.exp(4 * mc) * iqr

    # Identify outliers
    outlier_indices = []
    for idx, val in enumerate(values):
        if val < lower_bound or val > upper_bound:
            outlier_indices.append(idx)

    result = _build_result_dict(
        outlier_indices=outlier_indices,
        n_total=len(values),
        method='adjusted_boxplot',
        cutoff=cutoff,
        q1=q1, q3=q3, iqr=iqr, median=median,
        medcouple=float(mc),
        lower_bound=lower_bound, upper_bound=upper_bound
    )

    _print_detection_summary(result, verbose)
    return result


def hampel_outlier_detection_core(values: Union[np.ndarray, List[float]],
                                   cutoff: float = 3.5,
                                   verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    Hampel's rule for outlier detection.

    Step 1 - Calculate MAD (Equation 16 from reference):
        MAD = median(|x_i - median_sample|) × 1.483

    Step 2 - Calculate MAD-normalized values (Equation 17 from reference):
        x_{i,MAD-normalized} = |(x_i - median_sample) / MAD|

    Any MAD-normalized absolute value exceeding the cutoff (default 3.5)
    is declared an outlier.

    Parameters
    ----------
    values : array-like
        Input values (NaN values are removed internally)
    cutoff : float
        Threshold for MAD-normalized values (default 3.5)
    verbose : bool
        Print detection statistics

    Returns
    -------
    dict or None
        Result dictionary with outlier_indices and statistics,
        or None if insufficient data
    """
    values = _validate_and_clean_input(values)
    if values is None:
        return None

    # Consistency constant so MAD estimates population std dev for normal data
    CONSISTENCY_CONSTANT = 1.483

    # Step 1: Calculate MAD (Equation 16)
    median = float(np.median(values))
    median_abs_diff = float(np.median(np.abs(values - median)))
    MAD = CONSISTENCY_CONSTANT * median_abs_diff

    # Handle zero MAD (all values identical or nearly so)
    if median_abs_diff == 0:
        result = _build_result_dict(
            outlier_indices=[],
            n_total=len(values),
            method='hampel',
            cutoff=cutoff,
            median=median, mad=MAD, median_abs_diff=median_abs_diff
        )
        _print_detection_summary(result, verbose)
        return result

    # Step 2: Identify outliers using MAD-normalized values (Equation 17)
    outlier_indices = []
    for idx, val in enumerate(values):
        mad_normalized = abs(val - median) / MAD
        if mad_normalized > cutoff:
            outlier_indices.append(idx)

    # Calculate equivalent bounds for reporting
    lower_bound = median - cutoff * MAD
    upper_bound = median + cutoff * MAD

    result = _build_result_dict(
        outlier_indices=outlier_indices,
        n_total=len(values),
        method='hampel',
        cutoff=cutoff,
        median=median, mad=MAD, median_abs_diff=median_abs_diff,
        lower_bound=lower_bound, upper_bound=upper_bound
    )

    _print_detection_summary(result, verbose)
    return result


# ==============================================================================
# UNIFIED INTERFACE
# ==============================================================================

def detect_outliers(values: Union[np.ndarray, List[float]],
                    method: str = 'boxplot',
                    cutoff: float = None,
                    verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    Unified interface for outlier detection.

    Parameters
    ----------
    values : array-like
        1D array of numerical values (NaN values are filtered internally)
    method : str
        Detection method. One of:
        - 'boxplot': Standard IQR method
        - 'adjusted_boxplot': IQR with medcouple skewness correction
        - 'hampel': Hampel identifier
    cutoff : float, optional
        Method-specific threshold. If None, uses default for method:
        - boxplot: 1.5
        - adjusted_boxplot: 1.5
        - hampel: 3.5
    verbose : bool
        Print detection statistics

    Returns
    -------
    dict or None
        Returns None if insufficient data (<4 values after removing NaN).
        Otherwise returns dict with:
        - outlier_indices: List[int] - indices of outliers in input array
        - n_outliers: int - number of outliers detected
        - n_total: int - total values analyzed
        - outlier_percentage: float - percentage of values that are outliers
        - method: str - method name used
        - cutoff: float - cutoff value used
        - lower_bound: float - lower threshold (most methods)
        - upper_bound: float - upper threshold (most methods)
        - ... additional method-specific fields

    Raises
    ------
    ValueError
        If method is not recognized
    ImportError
        If method='adjusted_boxplot' and statsmodels is not available

    Examples
    --------
    >>> values = [1.0, 1.1, 1.2, 1.1, 5.0, 1.0, 1.2, 1.1]
    >>> result = detect_outliers(values, method='boxplot', cutoff=1.5)
    >>> print(result['outlier_indices'])
    [4]
    >>> print(result['n_outliers'])
    1
    """
    # Validate method
    method = method.lower()
    if method not in AVAILABLE_METHODS:
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Available methods: {AVAILABLE_METHODS}"
        )

    # Get default cutoff if not provided
    if cutoff is None:
        cutoff = DEFAULT_CUTOFFS[method]

    # Route to appropriate detection function
    if method == 'boxplot':
        return boxplot_outlier_detection_core(values, cutoff, verbose)

    elif method == 'adjusted_boxplot':
        return adjusted_boxplot_outlier_detection_core(values, cutoff, verbose)

    elif method == 'hampel':
        return hampel_outlier_detection_core(values, cutoff, verbose)

    else:
        # Should never reach here due to validation above
        raise ValueError(f"Unknown method: {method}")


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def get_default_cutoff(method: str) -> float:
    """
    Get the default cutoff value for a method.

    Parameters
    ----------
    method : str
        Method name

    Returns
    -------
    float
        Default cutoff value

    Raises
    ------
    ValueError
        If method is not recognized
    """
    method = method.lower()
    if method not in DEFAULT_CUTOFFS:
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Available methods: {AVAILABLE_METHODS}"
        )
    return DEFAULT_CUTOFFS[method]


def validate_method(method: str) -> bool:
    """
    Check if a method name is valid.

    Parameters
    ----------
    method : str
        Method name to validate

    Returns
    -------
    bool
        True if method is valid, False otherwise
    """
    return method.lower() in AVAILABLE_METHODS


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("OUTLIER DETECTION MODULE - TEST")
    print("=" * 70)

    # Test data with known outlier
    test_values = [1.0, 1.1, 1.2, 1.05, 1.15, 1.08, 1.12, 5.0]  # 5.0 is outlier

    print(f"\nTest data: {test_values}")
    print(f"Expected outlier: 5.0 (index 7)")

    print("\n" + "-" * 70)
    print("Testing all methods:")
    print("-" * 70)

    for method in AVAILABLE_METHODS:
        try:
            result = detect_outliers(test_values, method=method, verbose=False)
            if result:
                outlier_vals = [test_values[i] for i in result['outlier_indices']]
                print(f"\n{method}:")
                print(f"  Cutoff: {result['cutoff']}")
                print(f"  Outliers found: {result['n_outliers']} at indices {result['outlier_indices']}")
                print(f"  Outlier values: {outlier_vals}")
                if 'lower_bound' in result:
                    print(f"  Bounds: [{result['lower_bound']:.4f}, {result['upper_bound']:.4f}]")
            else:
                print(f"\n{method}: No result (insufficient data)")
        except ImportError as e:
            print(f"\n{method}: SKIPPED - {e}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
