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
HDX-MS EQUIVALENCE TESTING - DIRECT PERCENTILE METHOD
================================================================================

PURPOSE:
    Implements the Direct Percentile method for calculating deterministic
    acceptance limits for HDX-MS equivalence testing. 

ALGORITHM OVERVIEW:
    The method uses Šidák correction [Šidák, Z. Rectangular Confidence Regions for the Means of Multivariate Normal
    Distributions. J. Am. Stat. Assoc. 1967, 62 (318), 626–633. https://doi.org/10.1080/01621459.1967.10482935]
    to account for multiple testing (two criteria: |delta_D| and |CI_LP,UP|). The Šidák-corrected significance
    level is:

        alpha_s = 1 - (1 - alpha)^(1/m)

    where m=2 for the two tests, giving alpha_s ≈ 0.0253 for alpha = 0.05.
    This corresponds to the 97.47th percentile.

TWO CASES:
    1. SIMPLE CASE (Single Time Dominant):
       When all values at and above the percentile threshold come from ONE labeling
       time, we directly calculate the limit:
       - If r is integer: limit = (r+1)th largest value
       - If r is non-integer: limit = mean of floor(r)th and ceil(r)th largest

    2. COMPLEX CASE (Multiple Times Contribute):
       When multiple labeling times contribute to the upper tail, we use the
       combinatorial algorithm from the Supporting Information:
       - Calculate ratio = product(N_pairings - r_ti) / N_pairings^t
       - Iterate r up or down until ratio brackets (1 - alpha_s)
       - Final limit = average of bracketing values

INPUTS:
    - ExperimentContext (from experiment module)
    - Precomputed statistics DataFrame
    - Time points list

OUTPUTS:
    - Deterministic acceptance limits (delta_d_limit, eac)
    - Figures (optional)

DEPENDENCIES:
    - experiment (for ExperimentContext)
    - precompute (for build_max_arrays)
    - config_parser (for configuration access)
    - matplotlib (for figure generation)

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from math import floor, ceil

# Import from refactored modules
from precompute import build_max_arrays
from config_parser import get_config_value, get_config_bool
from report_formatting import format_report_limit
from plotting import (
    DIRECT_PERCENTILE_FIGSIZE,
    FIGURE_COLORS,
    DEFAULT_FORMAT,
    DEFAULT_DPI,
    DEFAULT_DIRECT_PERCENTILE_AXIS_TITLE_FONTSIZE,
    DEFAULT_DIRECT_PERCENTILE_AXIS_NUMBER_FONTSIZE,
    get_format_extension,
    get_figure_settings,
    configure_matplotlib_direct_percentile,
    save_figure,
    style_axes_publication,
    add_grid
)


def _compute_pairing_xticks(n_pairings: int, target_intervals: int = 6) -> List[int]:
    """
    Build readable x-axis ticks from the pairing count.
    """
    if n_pairings <= 0:
        return [0]

    if n_pairings <= 10:
        return list(range(0, n_pairings + 1))

    raw_step = n_pairings / max(target_intervals, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step_candidates = [multiplier * magnitude for multiplier in (1, 2, 5, 10)]
    step = min(step_candidates, key=lambda candidate: abs(candidate - raw_step))

    ticks = list(range(0, (n_pairings // step) * step + 1, step))
    if ticks[-1] != n_pairings:
        ticks.append(n_pairings)

    return ticks


_R_INTEGER_ABS_TOL = 1e-9


def _snap_near_integer(value: float, abs_tol: float = _R_INTEGER_ABS_TOL) -> Tuple[bool, int]:
    """
    Return whether ``value`` is close enough to an integer to treat it as exact.

    Returns the snapped integer alongside the boolean so call sites can share
    the same tolerance rule without duplicating rounding logic.
    """
    snapped = round(value)
    return math.isclose(value, snapped, abs_tol=abs_tol), snapped

# Type hint for ExperimentContext without circular import
if TYPE_CHECKING:
    from experiment import ExperimentContext


# ==============================================================================
# DIRECT PERCENTILE ANALYZER CLASS
# ==============================================================================

class DirectPercentileAnalyzer:
    """
    Implements the Direct Percentile method for deterministic acceptance limits.

    Parameters
    ----------
    context : ExperimentContext
        Experiment context with all calculated parameters
    precomputed_df : pd.DataFrame
        Precomputed statistics DataFrame
    time_points : list
        List of labeling times
    config : dict, optional
        Configuration dictionary (for output settings)
    verbose : bool, optional
        Print progress messages 

    Attributes
    ----------
    context : ExperimentContext
        Stored experiment context
    n_times : int
        Number of labeling times
    n_pairings : int
        Number of replicate pairings (from context)
    alpha : float
        Family-wise significance level
    alpha_s : float
        Šidák-corrected significance level
    max_delta_d : np.ndarray
        Pre-collapsed max |delta_D| values, shape (n_times, n_pairings)
    max_ci_lp_up : np.ndarray
        Pre-collapsed max |CI_LP,UP| values, shape (n_times, n_pairings)
    """

    def __init__(self, context: 'ExperimentContext',
                 precomputed_df: pd.DataFrame,
                 time_points: List[float],
                 config: Dict[str, Any] = None,
                 verbose: bool = None):
        """
        Initialize the analyzer.

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

        # Create time labels for figures (t1, t2, etc.)
        self.time_labels = [f'$t_{i+1}$' for i in range(self.n_times)]

        # Get output directory from config
        self.output_dir = get_config_value(self.config, 'OUTPUT_SETTINGS', 'output_directory', './outputs')

        # Get parameters not from config
        self.alpha = context.alpha
        self.n_pairings = context.n_pairings
        self.m = context.m  # Number of tests (SIDAK_M_PARAMETER)
        self.alpha_s = context.alpha_s  # Šidák-corrected alpha

        # Build max arrays: shape (n_times, n_pairings)
        self.max_delta_d, self.max_ci_lp_up = build_max_arrays(precomputed_df, self.time_points)

        self._print(f"Target percentile considering Šidák-correction: {context.percentile:.4f}th")

    def _print(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _calculate_limit_for_array(self, max_array: np.ndarray, metric_name: str) -> Dict[str, Any]:
        """
        Calculate the acceptance limit for a single metric (delta_d or ci_lp_up).

        Parameters
        ----------
        max_array : np.ndarray
            Shape (n_times, n_pairings) containing max values per time-pairing
        metric_name : str
            Name for reporting ('|delta_D|' or '|CI_LP,UP|')

        Returns
        -------
        dict
            Results including limit value and algorithm details
        """
        # Flatten and create array with time indices
        values_with_times = []
        for t_idx in range(self.n_times):
            for p_idx in range(self.n_pairings):
                val = max_array[t_idx, p_idx]
                if not np.isnan(val):
                    values_with_times.append((val, t_idx))

        # Sort by value descending (largest first)
        values_with_times.sort(key=lambda x: x[0], reverse=True)
        n_total = len(values_with_times)

        # Calculate r (number of values in upper tail)
        r_exact = self.n_pairings * self.alpha_s
        if r_exact < 1:
            min_pairings = math.ceil(1 / self.alpha_s)
            raise ValueError(
                f"Direct percentile requires r = n_pairings * alpha_s >= 1, got "
                f"n_pairings={self.n_pairings}, alpha_s={self.alpha_s:.12f}, r={r_exact:.6f}. "
                f"Increase pairings to at least {min_pairings} for this alpha."
            )

        self._print(f"  r = N_pairings × alpha_s = {self.n_pairings} × {self.alpha_s:.6f} = {r_exact:.4f}")

        # Check if all top r+1 values (at and above percentile) come from single time
        # Use int(r_exact) + 1, not ceil(r_exact), because ceil(7.0) = 7 but we need 8
        r_check = int(r_exact) + 1
        top_r_times = [values_with_times[i][1] for i in range(min(r_check, n_total))]
        unique_times_in_top_r = set(top_r_times)

        if len(unique_times_in_top_r) == 1:
            # SIMPLE CASE: Single time dominant
            dominant_time = self.time_points[list(unique_times_in_top_r)[0]]
            self._print(f"Values originating from a single labeling time: {dominant_time}s")

            result = self._simple_method(values_with_times, r_exact, metric_name)
            result['method'] = 'simple'
            result['dominant_time'] = dominant_time
        else:
            # COMPLEX CASE: Multiple times contribute
            self._print(f"Values originate from multiple labeling times - using combinatorial method")

            result = self._combinatorial_method(values_with_times, r_exact, metric_name)
            result['method'] = 'combinatorial'

        return result

    def _simple_method(self, values_with_times: List[Tuple[float, int]],
                       r_exact: float, metric_name: str) -> Dict[str, Any]:
        """
        Simple method when single labeling time dominates upper tail.

        If r is integer: limit = (r+1)th largest value
        If r is non-integer: limit = mean of floor(r)th and ceil(r)th largest
        """
        r_floor = floor(r_exact)
        r_ceil = ceil(r_exact)
        is_integer_like, r_snapped = _snap_near_integer(r_exact)

        if is_integer_like:  # r is integer
            limit_idx = r_snapped
            limit = values_with_times[limit_idx][0]
            self._print(f"  r is integer ({r_snapped}), limit = {r_snapped+1}th largest value = {limit:.4f}")
        else:
            idx_floor = r_floor - 1
            idx_ceil = r_ceil - 1
            val_floor = values_with_times[idx_floor][0]
            val_ceil = values_with_times[idx_ceil][0]
            limit = (val_floor + val_ceil) / 2
            self._print(f"  r is non-integer, limit = mean of {r_floor}th ({val_floor:.4f}) and {r_ceil}th ({val_ceil:.4f}) values")

        return {
            'limit': limit,
            'r_exact': r_exact,
            'r_floor': r_floor,
            'r_ceil': r_ceil
        }

    def _combinatorial_method(self, values_with_times: List[Tuple[float, int]],
                              r_exact: float, metric_name: str) -> Dict[str, Any]:
        """
        Combinatorial method from Supporting Information.

        Used when multiple labeling times contribute to the upper tail.
        """
        target = 1 - self.alpha_s

        r = int(r_exact)
        ratio, r_ti_dict = self._calculate_ratio(values_with_times, r)

        self._print(f"    Initial r={r}, ratio={ratio:.6f}, target={target:.6f}")

        iterations = []
        iterations.append({'r': r, 'ratio': ratio})

        if abs(ratio - target) < 1e-10:
            limit_idx = r
            limit = values_with_times[limit_idx][0]
            self._print(f"    Exact match at r={r}, limit = {limit:.4f}")

            return {
                'limit': limit,
                'r_final': r,
                'ratio_final': ratio,
                'iterations': iterations
            }

        if ratio < target:
            direction = -1
            self._print(f"    Ratio < target, decreasing r...")
        else:
            direction = 1
            self._print(f"    Ratio > target, increasing r...")

        max_iterations = 100
        prev_r = r
        prev_ratio = ratio
        bracket_found = False
        r_smaller = None
        r_larger = None

        for _ in range(max_iterations):
            r = r + direction

            if r < 0 or r >= len(values_with_times):
                break

            ratio, r_ti_dict = self._calculate_ratio(values_with_times, r)
            iterations.append({'r': r, 'ratio': ratio})

            if direction == -1:
                if ratio > target:
                    r_smaller = prev_r
                    r_larger = r
                    bracket_found = True
                    break
            else:
                if ratio < target:
                    r_smaller = r
                    r_larger = prev_r
                    bracket_found = True
                    break

            prev_r = r
            prev_ratio = ratio
        else:
            # Exhausting the search without a crossover means the intended
            # bracketing search failed and must not continue with invalid state.
            raise RuntimeError(
                f"{metric_name}: combinatorial bracketing search exhausted "
                f"{max_iterations} iterations without finding a valid bracket"
            )

        if not bracket_found or r_smaller is None or r_larger is None or r_smaller == r_larger:
            # Boundary exit without a crossover also means the manuscript-style
            # bracketing search failed and should raise clearly.
            raise RuntimeError(
                f"{metric_name}: combinatorial bracketing search failed "
                f"before finding a valid bracket (last r={r}, previous r={prev_r})"
            )

        val_smaller = values_with_times[r_smaller][0]
        val_larger = values_with_times[r_larger][0]
        limit = (val_smaller + val_larger) / 2

        self._print(f"    Bracketed between r={r_smaller} ({val_smaller:.4f}) and r={r_larger} ({val_larger:.4f})")
        self._print(f"    Limit = {limit:.4f}")

        return {
            'limit': limit,
            'r_smaller': r_smaller,
            'r_larger': r_larger,
            'val_smaller': val_smaller,
            'val_larger': val_larger,
            'iterations': iterations
        }

    def _calculate_ratio(self, values_with_times: List[Tuple[float, int]],
                         r: int) -> Tuple[float, Dict[int, int]]:
        """
        Calculate the combinatorial ratio for given r.

        ratio = product(N_pairings - r_ti) / N_pairings^t
        """
        r_ti = {t_idx: 0 for t_idx in range(self.n_times)}

        for i in range(min(r, len(values_with_times))):
            t_idx = values_with_times[i][1]
            r_ti[t_idx] += 1

        numerator = 1.0
        for t_idx in range(self.n_times):
            numerator *= (self.n_pairings - r_ti[t_idx])

        denominator = self.n_pairings ** self.n_times

        ratio = numerator / denominator

        return ratio, r_ti

    # ==========================================================================
    # FIGURE GENERATION
    # ==========================================================================

    def _configure_matplotlib(self) -> None:
        """
        Configure matplotlib for figures.
        Uses centralized configuration from plotting module.
        """
        configure_matplotlib_direct_percentile()

    def _calculate_per_time_thresholds(self, max_array: np.ndarray) -> np.ndarray:
        """Calculate per-time thresholds for supported direct-percentile inputs."""
        r_exact = self.n_pairings * self.alpha_s
        if r_exact < 1:
            min_pairings = math.ceil(1 / self.alpha_s)
            raise ValueError(
                f"Direct percentile requires r = n_pairings * alpha_s >= 1, got "
                f"n_pairings={self.n_pairings}, alpha_s={self.alpha_s:.12f}, r={r_exact:.6f}. "
                f"Increase pairings to at least {min_pairings} for this alpha."
            )
        r_floor = floor(r_exact)
        r_ceil = ceil(r_exact)
        is_integer_like, r_snapped = _snap_near_integer(r_exact)

        thresholds = np.zeros(self.n_times)

        for t in range(self.n_times):
            sorted_vals = np.sort(max_array[t, :])

            if is_integer_like:
                threshold = sorted_vals[self.n_pairings - r_snapped - 1]
            else:
                val_floor = sorted_vals[self.n_pairings - r_floor]
                val_ceil = sorted_vals[self.n_pairings - r_ceil]
                threshold = (val_floor + val_ceil) / 2

            thresholds[t] = threshold

        return thresholds

    def plot_ranked_values(self,
                           values_2d: np.ndarray,
                           final_limit: float,
                           y_label: str,
                           x_label: str,
                           output_path: str = None,
                           figsize: Tuple[float, float] = None,
                           dpi: int = None,
                           colors: List[str] = None,
                           show_plot: bool = True,
                           output_format: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create ranked value plot with percentile threshold line.
        """
        self._configure_matplotlib()

        # Get settings from config or use defaults
        settings = get_figure_settings(self.config, 'direct_percentile') if self.config else {
            'figsize': DIRECT_PERCENTILE_FIGSIZE,
            'axis_title_fontsize': DEFAULT_DIRECT_PERCENTILE_AXIS_TITLE_FONTSIZE,
            'axis_number_fontsize': DEFAULT_DIRECT_PERCENTILE_AXIS_NUMBER_FONTSIZE,
            'exact_figsize': True,
            'dpi': DEFAULT_DPI
        }

        axis_title_fontsize = settings['axis_title_fontsize']
        axis_number_fontsize = settings['axis_number_fontsize']
        exact_figsize = settings.get('exact_figsize', True)
        if dpi is None:
            dpi = settings.get('dpi', DEFAULT_DPI)

        T, NP = values_2d.shape

        if colors is None:
            colors = FIGURE_COLORS[:T]

        if figsize is None:
            figsize = settings['figsize']

        fig, ax = plt.subplots(figsize=figsize)

        # Legend order: increasing time index (t1, t2, t3, t4, t5, t6)
        time_order = list(range(T))

        # Find threshold crossing position (use time with highest max value)
        max_vals_per_time = [np.max(values_2d[t, :]) for t in range(T)]
        dominant_time_idx = int(np.argmax(max_vals_per_time))
        sorted_dominant = np.sort(values_2d[dominant_time_idx, :])
        crossing_indices = np.where(sorted_dominant >= final_limit)[0]
        threshold_rank = crossing_indices[0] + 1 if len(crossing_indices) > 0 else NP

        # Plot each labeling time
        lines = []
        labels = []
        for t in time_order:
            sorted_vals = np.sort(values_2d[t, :])
            ranks = np.arange(1, NP + 1)

            line, = ax.plot(ranks, sorted_vals, color=colors[t],
                           label=self.time_labels[t], linewidth=1.0)
            lines.append(line)
            labels.append(self.time_labels[t])

        # Add grid
        add_grid(ax)

        # Threshold lines
        ax.axhline(y=final_limit, color='black', linestyle='-', linewidth=1.0, zorder=10)
        ax.axvline(x=threshold_rank, color='black', linestyle='--', linewidth=1.0, zorder=10)

        # Axes
        ax.set_xlim(0, NP)
        ax.set_xticks(_compute_pairing_xticks(NP))
        ax.set_xlabel(x_label, fontsize=axis_title_fontsize)
        ax.set_ylabel(y_label, fontsize=axis_title_fontsize)
        ax.tick_params(axis='both', labelsize=axis_number_fontsize, direction='out')

        # Legend
        legend = ax.legend(
            lines, labels,
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            frameon=True,
            fancybox=False,
            edgecolor='black',
            framealpha=1.0,
            handlelength=1.2,
            handletextpad=0.4,
            borderpad=0.4,
            labelspacing=0.3
        )
        legend.get_frame().set_linewidth(0.5)

        # Apply consistent spine styling (keep all spines visible for this plot type)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        plt.tight_layout()

        if output_path:
            save_figure(fig, output_path, output_format=output_format, dpi=dpi,
                       figure_type='direct_percentile', exact_size=exact_figsize)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    def generate_delta_d_figure(self,
                                delta_d_limit: float,
                                output_path: str = None,
                                figsize: Tuple[float, float] = None,
                                show_plot: bool = True,
                                output_format: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """Generate Figure: |ΔD|_max ranked values plot."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = get_format_extension(output_format)
            output_path = os.path.join(self.output_dir, f"direct_percentile_deltaD_{timestamp}{ext}")

        return self.plot_ranked_values(
            values_2d=self.max_delta_d,
            final_limit=delta_d_limit,
            y_label=r'$|\Delta D|_{\mathrm{max}}$ (Da)',
            x_label=r'$|\Delta D|_{\mathrm{max}}$ rank',
            output_path=output_path,
            figsize=figsize,
            show_plot=show_plot,
            output_format=output_format
        )

    def generate_ci_figure(self,
                           eac_limit: float,
                           output_path: str = None,
                           figsize: Tuple[float, float] = None,
                           show_plot: bool = True,
                           output_format: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """Generate Figure: |CI_LP,UP|_max ranked values plot."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = get_format_extension(output_format)
            output_path = os.path.join(self.output_dir, f"direct_percentile_CI_{timestamp}{ext}")

        return self.plot_ranked_values(
            values_2d=self.max_ci_lp_up,
            final_limit=eac_limit,
            y_label=r'$|\mathrm{CI}_{\mathrm{LP, UP}}|_{\mathrm{max}}$ (Da)',
            x_label=r'$|\mathrm{CI}_{\mathrm{LP, UP}}|_{\mathrm{max}}$ rank',
            output_path=output_path,
            figsize=figsize,
            show_plot=show_plot,
            output_format=output_format
        )

    def calculate_acceptance_limits(self,
                                     generate_figures: bool = None,
                                     save_figures: bool = None,
                                     show_plots: bool = False,
                                     output_format: str = None) -> Dict[str, Any]:
        """
        Calculate acceptance limits for both |delta_D| and |CI_LP,UP|.

        Parameters
        ----------
        generate_figures : bool, optional
            Generate ranked value figures. If None, reads from config.
        save_figures : bool, optional
            Save figures to files. If None, reads from config.
        show_plots : bool
            Display the plots interactively
        output_format : str, optional
            Figure format: 'tiff' (default), 'png', or 'pdf'.
            If None, reads from config.

        Returns
        -------
        dict
            Results containing limits, method details, and figure paths
        """
        self._print("Calculating ΔD_limit:")
        delta_d_result = self._calculate_limit_for_array(self.max_delta_d, '|delta_D|')

        self._print("Calculating EAC:")
        ci_result = self._calculate_limit_for_array(self.max_ci_lp_up, '|CI_LP,UP|')

        results = {
            'delta_d_limit': delta_d_result['limit'],
            'eac': ci_result['limit'],
            'alpha': self.alpha,
            'alpha_s': self.alpha_s,
            'percentile': self.context.percentile,
            'n_pairings': self.n_pairings,
            'n_times': self.n_times,
            'delta_d_details': delta_d_result,
            'ci_details': ci_result,
            'figure_paths': []
        }

        print()  # Blank line before Limits:
        self._print(f"Limits:")
        self._print(
            f"EAC = {format_report_limit(results['eac'], self.config)} Da    "
            f"|ΔD|_limit = {format_report_limit(results['delta_d_limit'], self.config)} Da"
        )
        print()  # Blank line after limits

        # Handle figure generation
        if generate_figures is None:
            generate_figures = get_config_bool(self.config, 'OUTPUT_SETTINGS', 'generate_percentile_figures', True)
        if save_figures is None:
            save_figures = get_config_bool(self.config, 'OUTPUT_SETTINGS', 'save_plots', True)

        # Get output format from config if not specified
        if output_format is None:
            output_format = get_config_value(self.config, 'OUTPUT_SETTINGS', 'figure_format', DEFAULT_FORMAT)

        if generate_figures:
            if save_figures and not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = get_format_extension(output_format)
            delta_d_path = os.path.join(self.output_dir, f"direct_percentile_deltaD_{timestamp}{ext}") if save_figures else None
            self.generate_delta_d_figure(
                delta_d_limit=results['delta_d_limit'],
                output_path=delta_d_path,
                show_plot=show_plots,
                output_format=output_format
            )
            if delta_d_path:
                results['figure_paths'].append(delta_d_path)

            ci_path = os.path.join(self.output_dir, f"direct_percentile_CI_{timestamp}{ext}") if save_figures else None
            self.generate_ci_figure(
                eac_limit=results['eac'],
                output_path=ci_path,
                show_plot=show_plots,
                output_format=output_format
            )
            if ci_path:
                results['figure_paths'].append(ci_path)

            # Figure paths are returned in results; main.py handles printing

        return results


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    print("DIRECT PERCENTILE MODULE - TEST")
    print("\nThis module requires precomputed data to run.")
    print("Run main.py to execute the full pipeline.")
