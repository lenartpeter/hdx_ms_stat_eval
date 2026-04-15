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
HDX-MS EQUIVALENCE TESTING - PARTITIONED LIMITS METHOD
================================================================================

PURPOSE:
    Implements the Partitioned Limits method for HDX-MS equivalence
    testing.

Partitioning options:
    1. LABELING TIME: One partition per labeling time point
       Each peptide-time measurement is compared against the limit for its time.

    2. PEPTIDE LENGTH: Partitions based on amino acid count
       Partition width specified in config (e.g., 6 means partitions of 6 amino acids each)

    3. STANDARD ERROR: Partitions based on SE values
       Number of equal-width partitions specified in config

LIMIT CALCULATION:
    For each partition, the limit is simply the MAXIMUM value of |delta_D| or
    |CI_LP,UP| that falls within that partition. This is complete enumeration
    within each partition, not a percentile-based approach.

EVALUATION:
    When evaluating a candidate, each peptide-labeling time measurement is
    compared against the limit for the partition it falls into. A measurement
    passes if both |delta_D| and |CI_LP,UP| are within their respective limits

INPUTS:
    - ExperimentContext (from experiment module)
    - Precomputed statistics DataFrame
    - Configuration dictionary

OUTPUTS:
    - Stratified limits dictionaries
    - Evaluation results for candidates

DEPENDENCIES:
    - experiment (for ExperimentContext)
    - config_parser (for configuration access)
    - data_loader (for calculate_peptide_length)

================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from math import floor, ceil

# 
from config_parser import get_config_value, get_config_bool, get_config_list
from data_loader import calculate_peptide_length

# Type hint for ExperimentContext without circular import
if TYPE_CHECKING:
    from experiment import ExperimentContext


# ==============================================================================
# PARTITIONED LIMITS ANALYZER CLASS
# ==============================================================================

class PartitionedLimitsAnalyzer:
    """
    Implements partitioned (stratified) acceptance limits.

    Parameters
    ----------
    context : ExperimentContext
        Experiment context with all calculated parameters
    precomputed_df : pd.DataFrame
        Precomputed statistics DataFrame
    config : dict, optional
        Configuration dictionary (for stratification settings)
    verbose : bool, optional
        Print progress messages (default from context)

    Attributes
    ----------
    context : ExperimentContext
        Stored experiment context
    precomputed_df : pd.DataFrame
        Precomputed statistics with delta_d_abs column added
    time_points : list
        Unique labeling times
    peptide_lengths : list
        Unique peptide lengths
    se_min, se_max : float
        Range of standard error values
    """

    def __init__(self, context: 'ExperimentContext',
                 precomputed_df: pd.DataFrame,
                 config: Dict[str, Any] = None,
                 verbose: bool = None):
        """
        Initialize the analyzer.

        Parameters
        ----------
        context : ExperimentContext
            Experiment context with parameters
        precomputed_df : pd.DataFrame
            Precomputed statistics with columns:
            Time_sec, Sequence, Peptide_Length, delta_d, ci_lp_up_abs_max, se
        config : dict, optional
            Configuration dictionary
        verbose : bool, optional
            Print progress (default from context)
        """
        self.context = context
        self.config = config or {}
        self.verbose = verbose if verbose is not None else context.verbose

        # Make a copy and add absolute delta_d
        self.precomputed_df = precomputed_df.copy()
        self.precomputed_df['delta_d_abs'] = self.precomputed_df['delta_d'].abs()

        # Get unique values for partitioning
        self.time_points = sorted(self.precomputed_df['Time_sec'].unique().tolist())
        self.peptide_lengths = sorted(self.precomputed_df['Peptide_Length'].unique().tolist())
        self.se_min = self.precomputed_df['se'].min()
        self.se_max = self.precomputed_df['se'].max()

        self._print(f"Data range: {len(self.precomputed_df):,} records")
        self._print(f"  Labeling times: {self.time_points}")
        self._print(f"  Peptide lengths: {min(self.peptide_lengths)} - {max(self.peptide_lengths)} aa")
        self._print(f"  SE range: {self.se_min:.4f} - {self.se_max:.4f}")

    def _print(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    # ==========================================================================
    # STRATIFICATION BY LABELING TIME
    # ==========================================================================

    def calculate_limits_by_time(self) -> Dict[float, Dict[str, float]]:
        """
        Calculate limits stratified by labeling time.

        Returns
        -------
        dict
            {time: {'delta_d_limit': value, 'eac': value, 'n_records': count}}
        """
        limits = {}

        for time in self.time_points:
            time_data = self.precomputed_df[self.precomputed_df['Time_sec'] == time]

            delta_d_limit = time_data['delta_d_abs'].max()
            eac = time_data['ci_lp_up_abs_max'].max()

            limits[time] = {
                'delta_d_limit': float(delta_d_limit),
                'eac': float(eac),
                'n_records': len(time_data)
            }

        return limits

    # ==========================================================================
    # STRATIFICATION BY PEPTIDE LENGTH
    # ==========================================================================

    def calculate_limits_by_peptide_length(self, partition_width: int) -> Tuple[Dict[Tuple[int, int], Dict[str, float]], int, List[Tuple[int, int]]]:
        """
        Calculate limits stratified by peptide length.

        Parameters
        ----------
        partition_width : int
            Width of each partition in amino acids

        Returns
        -------
        tuple
            (limits_dict, n_empty_partitions, empty_partitions_list)
            limits_dict: {(partition_start, partition_end): {'delta_d_limit': value, 'eac': value}}
            n_empty_partitions: count of partitions with no precomputed data
            empty_partitions_list: list of (partition_start, partition_end) tuples that are empty
        """
        min_len = min(self.peptide_lengths)
        max_len = max(self.peptide_lengths)

        limits = {}
        empty_partitions = []

        # Create partitions
        partition_start = min_len
        while partition_start <= max_len:
            partition_end = partition_start + partition_width - 1
            partition_key = (partition_start, partition_end)

            # Get data in this partition
            partition_data = self.precomputed_df[
                (self.precomputed_df['Peptide_Length'] >= partition_start) &
                (self.precomputed_df['Peptide_Length'] <= partition_end)
            ]

            if len(partition_data) > 0:
                delta_d_limit = partition_data['delta_d_abs'].max()
                eac = partition_data['ci_lp_up_abs_max'].max()

                limits[partition_key] = {
                    'delta_d_limit': float(delta_d_limit),
                    'eac': float(eac),
                    'n_records': len(partition_data)
                }
            else:
                empty_partitions.append(partition_key)

            partition_start = partition_end + 1

        return limits, len(empty_partitions), empty_partitions

    def _get_peptide_length_partition_theoretical(self, length: int, partition_width: int) -> Tuple[int, int]:
        """
        Get the theoretical partition for a given peptide length based on partition_width.

        This returns the partition boundaries based purely on the partition scheme,
        regardless of whether the partition has data.
        """
        min_len = min(self.peptide_lengths)
        max_len = max(self.peptide_lengths)

        # Handle values below minimum - use first partition
        if length < min_len:
            return (min_len, min_len + partition_width - 1)

        # Handle values above maximum - use last partition
        if length > max_len:
            partition_start = min_len
            while partition_start + partition_width <= max_len:
                partition_start += partition_width
            return (partition_start, partition_start + partition_width - 1)

        # Find which partition this length falls into
        partition_start = min_len
        while partition_start <= max_len:
            partition_end = partition_start + partition_width - 1
            if partition_start <= length <= partition_end:
                return (partition_start, partition_end)
            partition_start = partition_end + 1

        # Fallback (shouldn't reach here)
        return (min_len, min_len + partition_width - 1)

    # ==========================================================================
    # STRATIFICATION BY STANDARD ERROR
    # ==========================================================================

    def _validate_se_partition_inputs(self, n_partitions: int) -> float:
        """Validate SE partition inputs and return the canonical partition width."""
        if n_partitions <= 0:
            raise ValueError("SE partitioning requires n_partitions > 0")

        if self.se_max <= 0:
            raise ValueError(
                "SE partitioning requires self.se_max > 0; "
                "a non-positive SE range is invalid for this method"
            )

        return self.se_max / n_partitions

    def _get_se_partition_bounds(self, partition_index: int, n_partitions: int) -> Tuple[float, float]:
        """Convert a partition index into canonical SE partition boundaries."""
        partition_width = self._validate_se_partition_inputs(n_partitions)

        if partition_index < 0 or partition_index >= n_partitions:
            raise ValueError(
                f"SE partition index {partition_index} is out of range for {n_partitions} partitions"
            )

        partition_low = partition_index * partition_width
        partition_high = (partition_index + 1) * partition_width
        return (partition_low, partition_high)

    def _get_se_partition_index(self, se_value: float, n_partitions: int) -> int:
        """
        Return the canonical SE partition index using explicit interval rules.

        Non-last bins use [low, high); the last bin uses [low, high]. Arithmetic
        truncation is avoided deliberately so lookup matches bin construction at
        floating-point boundaries.
        """
        self._validate_se_partition_inputs(n_partitions)

        if se_value <= 0:
            return 0

        if se_value >= self.se_max:
            return n_partitions - 1

        for partition_index in range(n_partitions):
            partition_low, partition_high = self._get_se_partition_bounds(partition_index, n_partitions)
            if partition_index == n_partitions - 1:
                if partition_low <= se_value <= partition_high:
                    return partition_index
            else:
                if partition_low <= se_value < partition_high:
                    return partition_index

        raise RuntimeError(
            f"SE value {se_value} did not map to a canonical partition despite valid inputs"
        )

    def calculate_limits_by_se(self, n_partitions: int) -> Tuple[Dict[Tuple[float, float], Dict[str, float]], int, List[Tuple[float, float]]]:
        """
        Calculate limits stratified by standard error.

        Parameters
        ----------
        n_partitions : int
            Number of equal-width partitions

        Returns
        -------
        tuple
            (limits_dict, n_empty_partitions, empty_partitions_list)
            limits_dict: {(partition_low, partition_high): {'delta_d_limit': value, 'eac': value}}
            n_empty_partitions: count of partitions with no precomputed data
            empty_partitions_list: list of (partition_low, partition_high) tuples that are empty
        """
        self._validate_se_partition_inputs(n_partitions)

        limits = {}
        empty_partitions = []

        for i in range(n_partitions):
            partition_low, partition_high = self._get_se_partition_bounds(i, n_partitions)
            partition_key = (partition_low, partition_high)

            # SE bin construction and lookup intentionally share the same
            # canonical helper logic: non-last bins use [low, high), the last
            # bin uses [low, high], and arithmetic truncation is avoided.
            if i == n_partitions - 1:
                partition_data = self.precomputed_df[
                    (self.precomputed_df['se'] >= partition_low) &
                    (self.precomputed_df['se'] <= partition_high)
                ]
            else:
                partition_data = self.precomputed_df[
                    (self.precomputed_df['se'] >= partition_low) &
                    (self.precomputed_df['se'] < partition_high)
                ]

            if len(partition_data) > 0:
                delta_d_limit = partition_data['delta_d_abs'].max()
                eac = partition_data['ci_lp_up_abs_max'].max()

                limits[partition_key] = {
                    'delta_d_limit': float(delta_d_limit),
                    'eac': float(eac),
                    'n_records': len(partition_data)
                }
            else:
                empty_partitions.append(partition_key)

        return limits, len(empty_partitions), empty_partitions

    def _get_se_partition_theoretical(self, se_value: float, n_partitions: int) -> Tuple[float, float]:
        """
        Get the theoretical partition for a given SE value based on n_partitions.

        This returns the partition boundaries based purely on the partition scheme,
        regardless of whether the partition has data.
        """
        partition_index = self._get_se_partition_index(se_value, n_partitions)
        return self._get_se_partition_bounds(partition_index, n_partitions)

    # ==========================================================================
    # EVALUATION
    # ==========================================================================

    def evaluate_candidate_by_time(self, candidate_results: List[Dict],
                                   limits: Dict[float, Dict[str, float]]) -> Dict[str, Any]:
        """
        Evaluate a candidate using time-stratified limits.

        Parameters
        ----------
        candidate_results : list
            List of evaluation results from evaluate_candidate() in main.py
            Each dict has: Sequence, HX time, delta_d, ci_abs_max, etc.
        limits : dict
            Time-stratified limits from calculate_limits_by_time()

        Returns
        -------
        dict
            Evaluation results
        """
        n_pass = 0
        n_fail = 0
        failed_peptides = []
        empty_partition_errors = []

        for result in candidate_results:
            time = result['HX time']
            delta_d_abs = abs(result['delta_d'])
            ci_abs_max = result['ci_abs_max']

            if time not in limits:
                empty_partition_errors.append({
                    'Sequence': result['Sequence'],
                    'HX time': time,
                    'error': f'No limit for time {time}s'
                })
                continue

            limit = limits[time]
            pass_delta_d = delta_d_abs < limit['delta_d_limit']
            pass_ci = ci_abs_max < limit['eac']

            if pass_delta_d and pass_ci:
                n_pass += 1
            else:
                n_fail += 1
                failed_peptides.append({
                    **result,
                    'limit_delta_d': limit['delta_d_limit'],
                    'limit_eac': limit['eac']
                })

        return {
            'n_total': n_pass + n_fail,
            'n_pass': n_pass,
            'n_fail': n_fail,
            'passed': n_fail == 0 and len(empty_partition_errors) == 0,
            'failed_peptides': failed_peptides,
            'empty_partition_errors': empty_partition_errors
        }

    def evaluate_candidate_by_peptide_length(self, candidate_results: List[Dict],
                                             limits: Dict[Tuple[int, int], Dict[str, float]],
                                             partition_width: int) -> Dict[str, Any]:
        """Evaluate a candidate using peptide length-stratified limits."""
        n_pass = 0
        n_fail = 0
        failed_peptides = []
        empty_partition_errors = []

        if not limits:
            for result in candidate_results:
                empty_partition_errors.append({
                    'Sequence': result['Sequence'],
                    'error': 'No peptide length limits available (all partitions empty)'
                })
            return {
                'n_total': 0,
                'n_pass': 0,
                'n_fail': 0,
                'passed': False,
                'failed_peptides': [],
                'empty_partition_errors': empty_partition_errors
            }

        partitions_sorted = sorted(limits.keys(), key=lambda x: x[0])
        lowest_partition = partitions_sorted[0]
        highest_partition = partitions_sorted[-1]

        for result in candidate_results:
            seq = result['Sequence']
            pep_len = calculate_peptide_length(seq)
            delta_d_abs = abs(result['delta_d'])
            ci_abs_max = result['ci_abs_max']

            # Handle values outside the precomputed range
            if pep_len < lowest_partition[0]:
                partition_key = lowest_partition
            elif pep_len > highest_partition[1]:
                partition_key = highest_partition
            else:
                partition_key = self._get_peptide_length_partition_theoretical(pep_len, partition_width)

                if partition_key not in limits:
                    empty_partition_errors.append({
                        'Sequence': seq,
                        'Peptide_Length': pep_len,
                        'partition': partition_key,
                        'error': f'Empty partition: length={pep_len} falls in partition ({partition_key[0]}, {partition_key[1]}) which has no reference data'
                    })
                    continue

            limit = limits[partition_key]
            pass_delta_d = delta_d_abs < limit['delta_d_limit']
            pass_ci = ci_abs_max < limit['eac']

            if pass_delta_d and pass_ci:
                n_pass += 1
            else:
                n_fail += 1
                failed_peptides.append({
                    **result,
                    'Peptide_Length': pep_len,
                    'partition': partition_key,
                    'limit_delta_d': limit['delta_d_limit'],
                    'limit_eac': limit['eac']
                })

        return {
            'n_total': n_pass + n_fail,
            'n_pass': n_pass,
            'n_fail': n_fail,
            'passed': n_fail == 0 and len(empty_partition_errors) == 0,
            'failed_peptides': failed_peptides,
            'empty_partition_errors': empty_partition_errors
        }

    def evaluate_candidate_by_se(self, candidate_results: List[Dict],
                                 limits: Dict[Tuple[float, float], Dict[str, float]],
                                 n_partitions: int,
                                 candidate_se_values: Dict[Tuple[str, float], float]) -> Dict[str, Any]:
        """
        Evaluate a candidate using SE-stratified limits.

        Parameters
        ----------
        candidate_results : list
            List of evaluation results
        limits : dict
            SE-stratified limits (only contains non-empty partitions)
        n_partitions : int
            Number of partitions used
        candidate_se_values : dict
            Mapping of (Sequence, Time) -> SE for the candidate
        """
        n_pass = 0
        n_fail = 0
        failed_peptides = []
        empty_partition_errors = []

        if not limits:
            for result in candidate_results:
                empty_partition_errors.append({
                    'Sequence': result['Sequence'],
                    'HX time': result['HX time'],
                    'error': 'No SE limits available (all partitions empty)'
                })
            return {
                'n_total': 0,
                'n_pass': 0,
                'n_fail': 0,
                'passed': False,
                'failed_peptides': [],
                'empty_partition_errors': empty_partition_errors
            }

        partitions_sorted = sorted(limits.keys(), key=lambda x: x[0])
        lowest_partition = partitions_sorted[0]
        highest_partition = partitions_sorted[-1]

        for result in candidate_results:
            seq = result['Sequence']
            time = result['HX time']
            delta_d_abs = abs(result['delta_d'])
            ci_abs_max = result['ci_abs_max']

            se_key = (seq, time)
            if se_key not in candidate_se_values:
                empty_partition_errors.append({
                    'Sequence': seq,
                    'HX time': time,
                    'error': 'No SE value available for this measurement'
                })
                continue

            se_value = candidate_se_values[se_key]

            # Handle values outside the precomputed range
            if se_value < lowest_partition[0]:
                partition_key = lowest_partition
            elif se_value > highest_partition[1]:
                partition_key = highest_partition
            else:
                partition_key = self._get_se_partition_theoretical(se_value, n_partitions)

                if partition_key not in limits:
                    empty_partition_errors.append({
                        'Sequence': seq,
                        'HX time': time,
                        'SE': se_value,
                        'partition': partition_key,
                        'error': f'Empty partition: SE={se_value:.6f} falls in partition ({partition_key[0]:.6f}, {partition_key[1]:.6f}) which has no reference data'
                    })
                    continue

            limit = limits[partition_key]
            pass_delta_d = delta_d_abs < limit['delta_d_limit']
            pass_ci = ci_abs_max < limit['eac']

            if pass_delta_d and pass_ci:
                n_pass += 1
            else:
                n_fail += 1
                failed_peptides.append({
                    **result,
                    'SE': se_value,
                    'partition': partition_key,
                    'limit_delta_d': limit['delta_d_limit'],
                    'limit_eac': limit['eac']
                })

        return {
            'n_total': n_pass + n_fail,
            'n_pass': n_pass,
            'n_fail': n_fail,
            'passed': n_fail == 0 and len(empty_partition_errors) == 0,
            'failed_peptides': failed_peptides,
            'empty_partition_errors': empty_partition_errors
        }

    # ==========================================================================
    # BATCH ANALYSIS (like Table S1)
    # ==========================================================================

    def run_all_configurations(self, candidate_results_dict: Dict[str, List[Dict]],
                               candidate_se_dict: Dict[str, Dict[Tuple[str, float], float]] = None) -> pd.DataFrame:
        """
        Run evaluation with all stratification configurations.

        Parameters
        ----------
        candidate_results_dict : dict
            {candidate_name: list of evaluation results}
        candidate_se_dict : dict, optional
            {candidate_name: {(seq, time): se_value}}

        Returns
        -------
        pd.DataFrame
            Results table similar to Supporting Info Table S1
        """
        results = []

        # Get configuration options
        do_time = get_config_bool(self.config, 'PARTITIONED_LIMITS', 'stratify_by_labeling_time', True)
        do_length = get_config_bool(self.config, 'PARTITIONED_LIMITS', 'stratify_by_peptide_length', True)
        do_se = get_config_bool(self.config, 'PARTITIONED_LIMITS', 'stratify_by_se', True)

        length_partitions = get_config_list(self.config, 'PARTITIONED_LIMITS', 'peptide_length_partition_sizes', [])
        length_partitions = [int(x) for x in length_partitions] if length_partitions else [6, 7, 8, 9, 10]

        se_partitions = get_config_list(self.config, 'PARTITIONED_LIMITS', 'se_n_partitions', [])
        se_partitions = [int(x) for x in se_partitions] if se_partitions else [70, 80, 90, 100]

        candidate_names = list(candidate_results_dict.keys())

        # Labeling time stratification
        if do_time:
            self._print("\nStratification by labeling time...")
            limits = self.calculate_limits_by_time()

            row = {
                'Stratification': 'Labeling time',
                'Parameter': len(self.time_points)
            }

            for cand_name in candidate_names:
                eval_result = self.evaluate_candidate_by_time(
                    candidate_results_dict[cand_name], limits
                )
                row[cand_name] = eval_result['n_fail']

            results.append(row)

        # Peptide length stratification
        if do_length:
            self._print("\nStratification by peptide length...")

            for partition_width in length_partitions:
                limits, n_empty, empty_list = self.calculate_limits_by_peptide_length(partition_width)

                row = {
                    'Stratification': 'Peptide length',
                    'Parameter': partition_width,
                    'Empty_partitions': n_empty
                }

                for cand_name in candidate_names:
                    eval_result = self.evaluate_candidate_by_peptide_length(
                        candidate_results_dict[cand_name], limits, partition_width
                    )
                    row[cand_name] = eval_result['n_fail']

                    if eval_result['empty_partition_errors']:
                        row[f'{cand_name}_errors'] = len(eval_result['empty_partition_errors'])

                results.append(row)

        # SE stratification
        if do_se and candidate_se_dict:
            self._print("\nStratification by standard error...")

            for n_partitions in se_partitions:
                limits, n_empty, empty_list = self.calculate_limits_by_se(n_partitions)

                row = {
                    'Stratification': 'Standard error',
                    'Parameter': n_partitions,
                    'Empty_partitions': n_empty
                }

                for cand_name in candidate_names:
                    if cand_name in candidate_se_dict:
                        eval_result = self.evaluate_candidate_by_se(
                            candidate_results_dict[cand_name], limits, n_partitions,
                            candidate_se_dict[cand_name]
                        )
                        row[cand_name] = eval_result['n_fail']

                        if eval_result['empty_partition_errors']:
                            row[f'{cand_name}_errors'] = len(eval_result['empty_partition_errors'])
                    else:
                        row[cand_name] = 'N/A (no SE data)'

                results.append(row)

        return pd.DataFrame(results)


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    print("PARTITIONED LIMITS MODULE - TEST")
    print("\nThis module requires precomputed data to run.")
    print("Run main.py to execute the full pipeline.")
