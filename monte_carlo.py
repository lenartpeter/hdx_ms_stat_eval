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
HDX-MS EQUIVALENCE TESTING - MONTE CARLO SIMULATION MODULE
================================================================================

PURPOSE:
    Implements Monte Carlo resampling simulation for HDX-MS equivalence testing.

ALGORITHM DESCRIPTION:
    For each simulated experiment:
      1. For each of T labeling times (e.g. 6):
         - Randomly select one of N possible replicate pairings (e.g., 280)
         - Look up the pre-collapsed max |delta_D| and max |CI_LP,UP| for this
           (time, pairing) combination (these are maximums across all peptides)
      2. Take the maximum across all T time points
      3. Record these two maximum values

    Repeat N times (e.g., 100 million) to build distributions.

OPTIMIZATION:
    The key insight is that for any given (time, pairing) combination, the maximum
    values across peptides are fix. So we:
      1. Pre-collapse the 314k values into (time, pairing) -> max arrays
      2. Store these as numpy arrays of shape (n_times, n_pairings)
      3. During simulation, use vectorized numpy operations

    This makes the simulation extremely fast (~10M simulations/second).

INPUTS:
    - ExperimentContext (from experiment module)
    - Precomputed statistics DataFrame
    - Time points list

OUTPUTS:
    - Distribution statistics (percentiles, mean, std)
    - Acceptance limits at Šidák-corrected percentile

DEPENDENCIES:
    - experiment (for ExperimentContext)
    - precompute (for build_max_arrays)
    - numpy, scipy (numerical operations)

================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from tqdm import tqdm
import time as time_module

#
from config_parser import get_config_value
from precompute import build_max_arrays
from statistics_util import calculate_percentile_from_alpha, calculate_sidak_alpha
from report_formatting import format_report_limit

# Type hint for ExperimentContext without circular import
if TYPE_CHECKING:
    from experiment import ExperimentContext


# ==============================================================================
# CONSTANTS
# ==============================================================================

DEFAULT_BATCH_SIZE = 100_000
DEFAULT_N_SIMULATIONS = 1_000_000


# ==============================================================================
# MONTE CARLO SIMULATOR CLASS
# ==============================================================================

class MonteCarloSimulator:
    """
    Implements Monte Carlo resampling simulation for HDX-MS equivalence testing.

    Parameters
    ----------
    context : ExperimentContext
        Experiment context with all calculated parameters
    precomputed_df : pd.DataFrame
        Precomputed statistics DataFrame
    time_points : list
        List of labeling times
    config : dict, optional
        Configuration dictionary (for additional settings)
    verbose : bool, optional
        Print progress messages (default from context)

    Attributes
    ----------
    context : ExperimentContext
        Stored experiment context
    n_times : int
        Number of labeling times
    n_pairings : int
        Number of replicate pairings (from context)
    max_delta_d_per_time_pairing : np.ndarray
        Pre-collapsed max |delta_D| values, shape (n_times, n_pairings)
    max_ci_lp_up_per_time_pairing : np.ndarray
        Pre-collapsed max |CI_LP,UP| values, shape (n_times, n_pairings)
    """

    def __init__(self, context: 'ExperimentContext',
                 precomputed_df: pd.DataFrame,
                 time_points: List[float],
                 config: Dict[str, Any] = None,
                 verbose: bool = None):
        """
        Initialize the Monte Carlo simulator.

        Parameters
        ----------
        context : ExperimentContext
            Experiment context with parameters
        precomputed_df : pd.DataFrame
            Precomputed statistics DataFrame
        time_points : list
            List of labeling times
        config : dict, optional
            Configuration dictionary
        verbose : bool, optional
            Print progress (default from context)
        """
        self.context = context
        self.config = config or {}
        self.verbose = verbose if verbose is not None else context.verbose

        self.time_points = sorted(time_points)
        self.n_times = len(self.time_points)
        self.n_pairings = context.n_pairings 

        self.time_to_idx = {t: idx for idx, t in enumerate(self.time_points)}
        self.pairing_ids = np.arange(self.n_pairings)

        # Pre-collapse statistics
        self._precompute_collapsed_statistics(precomputed_df)

        n_cells = self.max_delta_d_per_time_pairing.size
        memory_bytes = (self.max_delta_d_per_time_pairing.nbytes +
                       self.max_ci_lp_up_per_time_pairing.nbytes)
        self._print(f"Pre-collapsing statistics by (time, pairing) - Created arrays of shape ({self.n_times}, {self.n_pairings})")
        self._print(f"Total cells: {n_cells:,}, Memory usage: {memory_bytes/1024:.1f} KB")

        # Validate
        n_missing = np.isnan(self.max_delta_d_per_time_pairing).sum()
        if n_missing > 0:
            raise ValueError(f"Missing {n_missing} cells in precomputed data")

    def _print(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _precompute_collapsed_statistics(self, precomputed_df: pd.DataFrame) -> None:
        """Pre-compute max values for each (time, pairing) combination."""
        self.max_delta_d_per_time_pairing, self.max_ci_lp_up_per_time_pairing = \
            build_max_arrays(precomputed_df, self.time_points)

    def run_simulation(self,
                      n_simulations: int = None,
                      batch_size: int = None,
                      random_seed: Optional[int] = None,
                      dD_limit_abs: Optional[float] = None,
                      EAC: Optional[float] = None) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.

        Parameters
        ----------
        n_simulations : int, optional
            Number of simulations (default from context)
        batch_size : int, optional
            Batch size for vectorized processing (default from context)
        random_seed : int, optional
            Random seed for reproducibility (default from context)
        dD_limit_abs : float, optional
            External |delta_D| limit for pass rate calculation
        EAC : float, optional
            External EAC limit for pass rate calculation

        Returns
        -------
        dict
            Simulation results including limits and statistics
        """
        # Use context values as defaults
        if n_simulations is None:
            n_simulations = self.context.n_simulations
        if batch_size is None:
            batch_size = self.context.batch_size
        if random_seed is None:
            random_seed = self.context.random_seed

        self._print(f"Simulations: {n_simulations:,}, Batch size: {batch_size:,}, "
                   f"Random seed: {random_seed if random_seed else 'None (random)'}")

        rng = np.random.default_rng(random_seed)

        start_time = time_module.time()

        delta_d_all, ci_lp_up_all = self._run_vectorized_batched(
            n_simulations=n_simulations,
            batch_size=batch_size,
            rng=rng
        )

        elapsed = time_module.time() - start_time

        results = self._calculate_statistics(
            delta_d_all=delta_d_all,
            ci_lp_up_all=ci_lp_up_all,
            n_simulations=n_simulations,
            elapsed=elapsed,
            dD_limit_abs=dD_limit_abs,
            EAC=EAC,
            random_seed=random_seed
        )

        self._print_results(results)

        return results

    def _run_vectorized_batched(self,
                               n_simulations: int,
                               batch_size: int,
                               rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Run simulation using vectorized batched processing."""
        all_delta_d = []
        all_ci_lp_up = []

        n_batches = (n_simulations + batch_size - 1) // batch_size

        with tqdm(total=n_simulations, desc="Simulating", unit="sim",
                  disable=not self.verbose) as pbar:

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_simulations)
                batch_n = end_idx - start_idx

                # Generate random pairing selections
                all_pairing_ids = rng.integers(0, self.n_pairings, size=(self.n_times, batch_n))

                # Look up pre-computed values
                time_indices = np.arange(self.n_times)[:, np.newaxis]

                delta_d_per_time_sim = self.max_delta_d_per_time_pairing[time_indices, all_pairing_ids]
                ci_lp_up_per_time_sim = self.max_ci_lp_up_per_time_pairing[time_indices, all_pairing_ids]

                # Take maximum across time
                delta_d_batch = delta_d_per_time_sim.max(axis=0)
                ci_lp_up_batch = ci_lp_up_per_time_sim.max(axis=0)

                all_delta_d.append(delta_d_batch)
                all_ci_lp_up.append(ci_lp_up_batch)

                pbar.update(batch_n)

        return np.concatenate(all_delta_d), np.concatenate(all_ci_lp_up)

    def _calculate_statistics(self,
                             delta_d_all: np.ndarray,
                             ci_lp_up_all: np.ndarray,
                             n_simulations: int,
                             elapsed: float,
                             dD_limit_abs: Optional[float],
                             EAC: Optional[float],
                             random_seed: Optional[int]) -> Dict[str, Any]:
        """Calculate all statistics from simulation results."""

        # Use Šidák-corrected percentile from context
        percentile_value = self.context.percentile

        delta_d_limit = float(np.percentile(delta_d_all, percentile_value))
        eac_limit = float(np.percentile(ci_lp_up_all, percentile_value))

        pass_delta_d = delta_d_all <= delta_d_limit
        pass_ci = ci_lp_up_all <= eac_limit
        pass_both = pass_delta_d & pass_ci

        configured_pass_count = None
        configured_pass_pct = None
        configured_alpha = None
        configured_alpha_s = None
        configured_percentile = None
        if dD_limit_abs is not None and EAC is not None:
            configured_pass_both = ((delta_d_all < dD_limit_abs) &
                                    (ci_lp_up_all < EAC))
            configured_pass_count = int(configured_pass_both.sum())
            configured_pass_pct = 100.0 * configured_pass_count / n_simulations
            configured_alpha = (100.0 - configured_pass_pct) / 100.0
            configured_alpha_s = calculate_sidak_alpha(configured_alpha, self.context.m)
            configured_percentile = calculate_percentile_from_alpha(configured_alpha_s)

        results = {
            'delta_d_all': delta_d_all,
            'ci_lp_up_all': ci_lp_up_all,

            'delta_d_max': float(delta_d_all.max()),
            'delta_d_limit': delta_d_limit,
            'delta_d_mean': float(delta_d_all.mean()),
            'delta_d_median': float(np.median(delta_d_all)),
            'delta_d_std': float(delta_d_all.std()),

            'eac_max': float(ci_lp_up_all.max()),
            'eac_limit': eac_limit,
            'eac_mean': float(ci_lp_up_all.mean()),
            'eac_median': float(np.median(ci_lp_up_all)),
            'eac_std': float(ci_lp_up_all.std()),

            'joint_pct_pass_both': 100.0 * pass_both.sum() / n_simulations,
            'configured_dD': dD_limit_abs,
            'configured_EAC': EAC,
            'configured_joint_pass_count': configured_pass_count,
            'configured_joint_pass_pct': configured_pass_pct,
            'configured_alpha': configured_alpha,
            'configured_alpha_s': configured_alpha_s,
            'configured_percentile': configured_percentile,

            'n_simulations': n_simulations,
            'elapsed_seconds': elapsed,
            'simulations_per_second': n_simulations / elapsed if elapsed > 0 else 0,
            'random_seed': random_seed,
            'percentile_used': percentile_value,
            'alpha_s': self.context.alpha_s
        }

        return results

    def _print_results(self, results: Dict[str, Any]) -> None:
        """Print formatted simulation results."""
        percentile = results['percentile_used']

        self._print(f"Completed in {results['elapsed_seconds']:.1f} seconds")

        self._print(f"{'Statistic':<20} {'|ΔD|_max (Da)':<15} {'|CI_LP,UP|_max (Da)':<15}")
        self._print(f"{'-'*20} {'-'*15} {'-'*15}")
        self._print(f"{'Maximum':<20} {results['delta_d_max']:<15.4f} {results['eac_max']:<15.4f}")
        self._print(f"{'Mean':<20} {results['delta_d_mean']:<15.4f} {results['eac_mean']:<15.4f}")
        self._print(f"{'Median':<20} {results['delta_d_median']:<15.4f} {results['eac_median']:<15.4f}")
        self._print(f"{'Std Dev':<20} {results['delta_d_std']:<15.4f} {results['eac_std']:<15.4f}")
        delta_d_limit_str = format_report_limit(results['delta_d_limit'], self.config)
        eac_limit_str = format_report_limit(results['eac_limit'], self.config)

        self._print(f"{f'{percentile:.2f}th percentile':<20} {delta_d_limit_str:<15} {eac_limit_str:<15}")
        print()  # Blank line 
        self._print(f"Limits:")
        self._print(f"EAC = {eac_limit_str} Da    |ΔD|_limit = {delta_d_limit_str} Da")
        print()  # Blank line after limits
        self._print(f"Number of simulated resampled data passing both criteria (Joint pass rate): {results['joint_pct_pass_both']:.4f}%")
        if (results['configured_dD'] is not None and
                results['configured_EAC'] is not None):
            configured_dD_str = get_config_value(self.config, 'STATISTICAL_PARAMETERS', 'dD', '').strip()
            configured_EAC_str = get_config_value(self.config, 'STATISTICAL_PARAMETERS', 'EAC', '').strip()

            if not configured_dD_str:
                configured_dD_str = str(results['configured_dD'])
            if not configured_EAC_str:
                configured_EAC_str = str(results['configured_EAC'])

            print()
            self._print("-" * 40)
            self._print(
                "Preset threshold evaluation:"
            )
            self._print(
                f"dD = {configured_dD_str} Da    "
                f"EAC = {configured_EAC_str} Da"
            )
            self._print(
                "Simulated experiments passing both preset thresholds: "
                f"{results['configured_joint_pass_count']:,}/{results['n_simulations']:,} "
                f"({results['configured_joint_pass_pct']:.4f}%)"
            )
            self._print(
                f"Percentile: "
                f"{results['configured_percentile']:.4f}th"
            )


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    print("MONTE CARLO MODULE - TEST")
    print("\nThis module requires precomputed data to run.")
    print("Run main.py to execute the full pipeline.")
