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
HDX-MS EQUIVALENCE TESTING - RESAMPLING MODULE
================================================================================

PURPOSE:
    Resampling method of Hageman et al. [Hageman, T. S.; Wrigley, M. S.; Weis, D. D. 
    Statistical Equivalence Testing of Higher-Order Protein Structures with Differential 
    Hydrogen Exchange-Mass Spectrometry (HX-MS). Anal. Chem. 2021, 93 (18), 6980–6988. 
    https://doi.org/10.1021/acs.analchem.0c05279] for establishing acceptance
    limits in HDX-MS equivalence testing. 

MODES:
    Mode A: Single random pairing applied to all labeling times
            - Each simulation uses 1 pairing → n data points

    Mode B: Different random pairing per labeling time
            - Each simulation uses T pairings (one per time) → n data points

INPUTS:
    - ExperimentContext (from experiment module)
    - Precomputed statistics DataFrame
    - Data structure dictionary

OUTPUTS:
    - Resampling figure
    - Results dict with EAC, delta_D_limit, statistics
    - Optional CSV with detailed per-peptide analysis data

DEPENDENCIES:
    - experiment (for ExperimentContext)
    - config_parser (for configuration access)
    - matplotlib (for figure generation)

================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING

# 
from config_parser import get_config_value, get_config_bool, get_config_int, get_config_float
from report_formatting import format_report_limit
from plotting import (
    RESAMPLING_FIGSIZE,
    DEFAULT_FORMAT,
    DEFAULT_DPI,
    DEFAULT_RESAMPLING_AXIS_TITLE_FONTSIZE,
    DEFAULT_RESAMPLING_AXIS_NUMBER_FONTSIZE,
    DEFAULT_RESAMPLING_RESULT_TEXT_FONTSIZE,
    DEFAULT_RESAMPLING_DOT_SIZE,
    DEFAULT_RESAMPLING_CI_LINE_WIDTH,
    DEFAULT_RESAMPLING_LIMIT_LINE_WIDTH,
    get_format_extension,
    get_figure_settings,
    configure_matplotlib_publication,
    save_figure,
    style_axes_publication
)

# Type hint for ExperimentContext without circular import
if TYPE_CHECKING:
    from experiment import ExperimentContext


# ==============================================================================
# HAGEMAN RESAMPLER CLASS
# ==============================================================================

class HagemanResampler:
    """

    Parameters
    ----------
    context : ExperimentContext
        Experiment context with all calculated parameters
    config : dict
        Configuration dictionary
    precomputed_df : pd.DataFrame
        Precomputed statistics from Precomputer
    data_structure : dict
        Data structure with time_points, peptides, n_replicates
    verbose : bool, optional
        Print progress messages (default from context)

    Attributes
    ----------
    context : ExperimentContext
        Stored experiment context
    precomputed_df : pd.DataFrame
        Precomputed statistics
    n_pairings : int
        Number of available pairings (from context)
    time_points : list
        List of labeling times
    """

    def __init__(self,
                 context: 'ExperimentContext',
                 config: Dict[str, Any],
                 precomputed_df: pd.DataFrame,
                 data_structure: Dict[str, Any],
                 verbose: bool = None):
        """
        Initialize the resampler.

        Parameters
        ----------
        context : ExperimentContext
            Experiment context with parameters
        config : dict
            Configuration dictionary
        precomputed_df : pd.DataFrame
            Precomputed statistics DataFrame with columns:
            Time_sec, Sequence, Pairing_ID, delta_d, ci_lp, ci_up, ci_lp_up_abs_max
        data_structure : dict
            Dictionary containing:
            - time_points: list of labeling times
            - peptides: list of peptide sequences
            - n_replicates: number of replicates
        verbose : bool, optional
            Print progress (default from context)
        """
        self.context = context
        self.config = config
        self.precomputed_df = precomputed_df
        self.data_structure = data_structure
        self.verbose = verbose if verbose is not None else context.verbose

        # Extract structure
        self.time_points = data_structure['time_points']
        self.peptides = data_structure['peptides']
        self.n_times = len(self.time_points)
        self.n_peptides = len(self.peptides)

        # Get unique pairing IDs from precomputed data
        self.pairing_ids = self.precomputed_df['Pairing_ID'].unique()
        self.n_pairings = len(self.pairing_ids)

        # Get output directory from config
        self.output_dir = get_config_value(config, 'OUTPUT_SETTINGS', 'output_directory', './outputs')

        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self._print(f"HagemanResampler initialized:")
        self._print(f"  Time points: {self.n_times}")
        self._print(f"  Peptides: {self.n_peptides}")
        self._print(f"  Available pairings: {self.n_pairings}")
        self._print(f"  Data points per simulation: {self.n_times * self.n_peptides:,}")

    def _print(self, message: str) -> None:
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(message)

    # ==========================================================================
    # SIMULATION METHODS
    # ==========================================================================

    def _get_simulation_data_mode_a(self) -> pd.DataFrame:
        """
        Mode A: Single random pairing for all time points.

        Returns
        -------
        pd.DataFrame
            Subset of precomputed data for the selected pairing
        """
        pairing_id = np.random.choice(self.pairing_ids)
        mask = self.precomputed_df['Pairing_ID'] == pairing_id
        subset = self.precomputed_df[mask].copy()

        # Include all relevant columns for detailed analysis output
        columns = ['Time_sec', 'Sequence', 'Peptide_Length', 'Pairing_ID',
                   'delta_d', 'ci_lp', 'ci_up', 'ci_lp_up_abs_max',
                   'se', 'sd_ref', 'sd_sample']
        return subset[columns].reset_index(drop=True)

    def _get_simulation_data_mode_b(self) -> pd.DataFrame:
        """
        Mode B: Different random pairing per time point.

        Returns
        -------
        pd.DataFrame
            Combined data from different pairings per time
        """
        # Include all relevant columns for detailed analysis output
        columns = ['Time_sec', 'Sequence', 'Peptide_Length', 'Pairing_ID',
                   'delta_d', 'ci_lp', 'ci_up', 'ci_lp_up_abs_max',
                   'se', 'sd_ref', 'sd_sample']

        all_data = []

        for time_point in self.time_points:
            pairing_id = np.random.choice(self.pairing_ids)

            mask = ((self.precomputed_df['Time_sec'] == time_point) &
                    (self.precomputed_df['Pairing_ID'] == pairing_id))
            subset = self.precomputed_df[mask]

            if len(subset) > 0:
                all_data.append(subset[columns])

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame(columns=columns)

    # ==========================================================================
    # PLOTTING
    # ==========================================================================

    def plot_hageman_figure(self,
                            data: pd.DataFrame,
                            eac: float = None,
                            delta_d_limit: float = None,
                            panel_label: str = None,
                            figsize: Tuple[float, float] = None,
                            save_path: str = None,
                            show_plot: bool = True,
                            output_format: str = None) -> Tuple[plt.Figure, plt.Axes, Dict]:
        """
        Figure creation.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns: delta_d, ci_lp, ci_up
        eac : float, optional
            Equivalence Acceptance Criterion. If None, calculated from data.
        delta_d_limit : float, optional
            Delta D limit. If None, calculated from data.
        panel_label : str, optional
            Panel label (e.g., "A", "B", "C")
        figsize : tuple
            Figure size in inches. ACS max: 7 x 9.167
        save_path : str, optional
            Path to save figure
        show_plot : bool
            Whether to display the plot

        Returns
        -------
        tuple
            (fig, ax, stats_dict)
        """
        # Configure matplotlib 
        configure_matplotlib_publication()

        # Get settings from config or use defaults
        settings = get_figure_settings(self.config, 'resampling') if self.config else {
            'figsize': RESAMPLING_FIGSIZE,
            'axis_title_fontsize': DEFAULT_RESAMPLING_AXIS_TITLE_FONTSIZE,
            'axis_number_fontsize': DEFAULT_RESAMPLING_AXIS_NUMBER_FONTSIZE,
            'result_text_fontsize': DEFAULT_RESAMPLING_RESULT_TEXT_FONTSIZE,
            'dot_size': DEFAULT_RESAMPLING_DOT_SIZE,
            'ci_line_width': DEFAULT_RESAMPLING_CI_LINE_WIDTH,
            'limit_line_width': DEFAULT_RESAMPLING_LIMIT_LINE_WIDTH,
            'exact_figsize': True,
            'dpi': DEFAULT_DPI
        }

        axis_title_fontsize = settings['axis_title_fontsize']
        axis_number_fontsize = settings['axis_number_fontsize']
        result_text_fontsize = settings.get('result_text_fontsize', DEFAULT_RESAMPLING_RESULT_TEXT_FONTSIZE)
        dot_size = settings.get('dot_size', DEFAULT_RESAMPLING_DOT_SIZE)
        ci_line_width = settings.get('ci_line_width', DEFAULT_RESAMPLING_CI_LINE_WIDTH)
        limit_line_width = settings.get('limit_line_width', DEFAULT_RESAMPLING_LIMIT_LINE_WIDTH)
        exact_figsize = settings.get('exact_figsize', True)
        dpi = settings.get('dpi', DEFAULT_DPI)

        # Use default figsize if not provided
        if figsize is None:
            figsize = settings['figsize']

        # Sort by delta_d for ranking
        sorted_data = data.sort_values('delta_d').reset_index(drop=True)

        delta_d = sorted_data['delta_d'].values
        ci_lp = sorted_data['ci_lp'].values
        ci_up = sorted_data['ci_up'].values

        n_points = len(delta_d)

        # Calculate limits if not provided
        if eac is None:
            eac = max(np.abs(ci_lp).max(), np.abs(ci_up).max())
        if delta_d_limit is None:
            delta_d_limit = np.abs(delta_d).max()

        rank = np.arange(n_points)

        fig, ax = plt.subplots(figsize=figsize)

        # ======================================================================
        # FAST PLOTTING: Use LineCollection for CI bars (gray vertical lines)
        # ======================================================================
        segments = np.zeros((n_points, 2, 2))
        segments[:, 0, 0] = rank
        segments[:, 0, 1] = ci_lp
        segments[:, 1, 0] = rank
        segments[:, 1, 1] = ci_up

        lc = LineCollection(segments, colors='gray', linewidths=ci_line_width, alpha=0.7)
        ax.add_collection(lc)

        # Plot delta_D values (black dots)
        ax.scatter(rank, delta_d, s=dot_size, c='black', zorder=5)

        # Horizontal lines for ±EAC (solid gray)
        ax.axhline(y=eac, color='gray', linestyle='-', linewidth=limit_line_width)
        ax.axhline(y=-eac, color='gray', linestyle='-', linewidth=limit_line_width)

        # Horizontal lines for ±delta_D_limit (dashed black)
        ax.axhline(y=delta_d_limit, color='black', linestyle='--', linewidth=limit_line_width)
        ax.axhline(y=-delta_d_limit, color='black', linestyle='--', linewidth=limit_line_width)

        # Labels and formatting
        ax.set_xlabel(r'$\Delta D$ rank', fontsize=axis_title_fontsize, labelpad=10)
        ax.set_ylabel(r'$\Delta D$ (Da)', fontsize=axis_title_fontsize)
        ax.tick_params(axis='both', labelsize=axis_number_fontsize)

        # Hide top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlim(0, n_points)
        y_max = max(eac, np.abs(ci_lp).max(), np.abs(ci_up).max()) * 1.15
        ax.set_ylim(-y_max, y_max)

        # Panel label (upper left, bold)
        if panel_label:
            ax.text(0.01, 0.99, panel_label,
                    transform=ax.transAxes,
                    fontsize=14,
                    fontweight='bold',
                    verticalalignment='top',
                    horizontalalignment='left')

        # EAC and delta_D_limit annotation (above plot frame, right-aligned)
        # Using \mathrm{limit} to make "limit" non-italic (roman text) in subscript
        annotation_text = (
            f'EAC = {format_report_limit(eac, self.config)}    '
            f'$\\Delta D_{{\\mathrm{{limit}}}}$ = {format_report_limit(delta_d_limit, self.config)}'
        )
        ax.text(1.0, 1.02, annotation_text,
                transform=ax.transAxes,
                fontsize=result_text_fontsize,
                verticalalignment='bottom',
                horizontalalignment='right')

        plt.tight_layout()

        if save_path:
            actual_path = save_figure(fig, save_path, output_format=output_format,
                                      dpi=dpi, figure_type='resampling', exact_size=exact_figsize)
            self._print(f"  Figure saved: {actual_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        stats_dict = {
            'eac': eac,
            'delta_d_limit': delta_d_limit,
            'n_points': n_points
        }

        return fig, ax, stats_dict

    # ==========================================================================
    # MAIN RESAMPLING METHOD
    # ==========================================================================

    def run_resampling(self,
                       mode: str = 'B',
                       n_simulations: int = 5,
                       n_rounds: int = 1,
                       generate_plot: bool = True,
                       save_fig: bool = True,
                       save_results: bool = False,
                       show_plot: bool = True,
                       panel_label: str = None,
                       figsize: Tuple[float, float] = None,
                       output_format: str = None) -> Dict[str, Any]:
        """
        Run resampling experiment.

        Parameters
        ----------
        mode : str
            'A' or 'B' - simulation mode
        n_simulations : int
            Number of simulations to pool per round (default 5)
        n_rounds : int
            Number of rounds to run (default 1)
        generate_plot : bool
            Generate Hageman-style figure (one per round)
        save_fig : bool
            Save figure to file
        save_results : bool
            Save results to Excel
        show_plot : bool
            Display the plot
        panel_label : str, optional
            Panel label (e.g., "A", "B")
        figsize : tuple
            Figure size
        output_format : str, optional
            Figure format: 'tiff' (default), 'png', or 'pdf'

        Returns
        -------
        dict
            Results containing limits and statistics
        """
        mode = mode.upper()
        if mode not in ['A', 'B']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'A' or 'B'.")

        self._print(f"\n{'='*70}")
        self._print(f"HAGEMAN RESAMPLING - MODE {mode}")
        self._print(f"{'='*70}")
        self._print(f"  Configuration: {n_simulations} simulations × {n_rounds} round(s)")

        if mode == 'A':
            sim_func = self._get_simulation_data_mode_a
            mode_desc = "Single pairing for all time points"
        else:
            sim_func = self._get_simulation_data_mode_b
            mode_desc = "Different pairing per time point"

        self._print(f"  Mode description: {mode_desc}")

        eac_per_round = []
        delta_d_limit_per_round = []
        per_round_results = []
        figure_paths = []

        for round_num in range(n_rounds):
            if n_rounds > 1:
                self._print(f"\n  --- Round {round_num + 1}/{n_rounds} ---")

            # Pool data from n_simulations
            pooled_data_list = []
            for _ in range(n_simulations):
                sim_data = sim_func()
                pooled_data_list.append(sim_data)

            pooled_data = pd.concat(pooled_data_list, ignore_index=True)

            eac = pooled_data['ci_lp_up_abs_max'].max()
            delta_d_limit = np.abs(pooled_data['delta_d']).max()
            n_points = len(pooled_data)

            eac_per_round.append(eac)
            delta_d_limit_per_round.append(delta_d_limit)

            self._print(f"    Data points: {n_points:,}")
            print()  # Blank line before Limits
            self._print(f"    Limits:")
            self._print(
                f"    EAC = {format_report_limit(eac, self.config)} Da    "
                f"|ΔD|_limit = {format_report_limit(delta_d_limit, self.config)} Da"
            )

            fig_path = None
            if generate_plot:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                round_suffix = f"_round{round_num + 1}" if n_rounds > 1 else ""
                ext = get_format_extension(output_format)
                fig_filename = f"hageman_mode_{mode}_{n_simulations}sim{round_suffix}_{timestamp}{ext}"
                fig_path = os.path.join(self.output_dir, fig_filename) if save_fig else None

                round_panel = panel_label
                if round_panel is None and n_rounds > 1:
                    round_panel = chr(ord('A') + round_num)

                fig, ax, plot_stats = self.plot_hageman_figure(
                    data=pooled_data,
                    eac=eac,
                    delta_d_limit=delta_d_limit,
                    panel_label=round_panel,
                    figsize=figsize,
                    save_path=fig_path,
                    show_plot=show_plot,
                    output_format=output_format
                )
                figure_paths.append(fig_path)

            per_round_results.append({
                'round': round_num + 1,
                'n_points': n_points,
                'eac': float(eac),
                'delta_d_limit': float(delta_d_limit),
                'figure_path': fig_path,
                'pooled_data': pooled_data
            })

        eac_arr = np.array(eac_per_round)
        delta_d_arr = np.array(delta_d_limit_per_round)

        results = {
            'mode': mode,
            'mode_description': mode_desc,
            'n_simulations': n_simulations,
            'n_rounds': n_rounds,
            'per_round_results': per_round_results,
            'eac_per_round': eac_per_round,
            'delta_d_limit_per_round': delta_d_limit_per_round,
            'eac_max': float(eac_arr.max()),
            'eac_mean': float(eac_arr.mean()),
            'eac_std': float(eac_arr.std()) if n_rounds > 1 else 0.0,
            'eac_median': float(np.median(eac_arr)),
            'delta_d_limit_max': float(delta_d_arr.max()),
            'delta_d_limit_mean': float(delta_d_arr.mean()),
            'delta_d_limit_std': float(delta_d_arr.std()) if n_rounds > 1 else 0.0,
            'delta_d_limit_median': float(np.median(delta_d_arr)),
            'figure_paths': figure_paths
        }

        if n_rounds > 1:
            self._print(f"\n  {'='*50}")
            self._print(f"  SUMMARY ACROSS {n_rounds} ROUNDS")
            self._print(f"  {'='*50}")
            self._print(f"  EAC:        max={results['eac_max']:.4f}, mean={results['eac_mean']:.4f}, std={results['eac_std']:.4f}")
            self._print(f"  ΔD_limit:   max={results['delta_d_limit_max']:.4f}, mean={results['delta_d_limit_mean']:.4f}, std={results['delta_d_limit_std']:.4f}")

        if save_results:
            self._save_analysis_csv(results)

        self._print(f"{'='*70}\n")

        return results

    def _save_analysis_csv(self, results: Dict[str, Any]) -> None:
        """
        Save detailed analysis results to CSV file.

        The CSV contains per-peptide statistical data from the resampling analysis,
        allowing users to review calculations and understand how acceptance limits
        were derived.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"resampling_mode_{results['mode']}_{results['n_simulations']}sim_{results['n_rounds']}rounds_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)

        # Collect all pooled data from each round
        all_data = []
        for round_result in results['per_round_results']:
            round_df = round_result['pooled_data'].copy()
            round_df['Round'] = round_result['round']
            round_df['dD_abs'] = round_df['delta_d'].abs()
            all_data.append(round_df)

        # Concatenate all rounds
        combined_df = pd.concat(all_data, ignore_index=True)

        # Reorder columns for clarity
        column_order = ['Round', 'Time_sec', 'Sequence', 'Peptide_Length', 'Pairing_ID',
                        'delta_d', 'dD_abs', 'ci_lp', 'ci_up', 'ci_lp_up_abs_max',
                        'se', 'sd_ref', 'sd_sample']
        combined_df = combined_df[column_order]

        # Save to CSV
        combined_df.to_csv(filepath, index=False)

        self._print(f"  Analysis data saved: {filepath}")

    # ==========================================================================
    # COMPARISON WITH COMPLETE ENUMERATION
    # ==========================================================================

    def get_complete_enumeration_limits(self) -> Dict[str, float]:
        """Get limits from complete enumeration (all precomputed values)."""
        eac = self.precomputed_df['ci_lp_up_abs_max'].max()
        delta_d_limit = np.abs(self.precomputed_df['delta_d']).max()
        n_points = len(self.precomputed_df)

        return {
            'eac': float(eac),
            'delta_d_limit': float(delta_d_limit),
            'n_points': n_points
        }

    def plot_complete_enumeration(self,
                                   panel_label: str = None,
                                   figsize: Tuple[float, float] = None,
                                   save_fig: bool = True,
                                   show_plot: bool = True,
                                   output_format: str = None) -> Dict[str, Any]:
        """Generate figure for complete enumeration data."""
        self._print(f"\n{'='*70}")
        self._print("COMPLETE ENUMERATION")
        self._print(f"{'='*70}")

        limits = self.get_complete_enumeration_limits()

        self._print(f"  Total data points: {limits['n_points']:,}")
        self._print(f"  EAC = {format_report_limit(limits['eac'], self.config)} Da")
        self._print(f"  ΔD_limit = {format_report_limit(limits['delta_d_limit'], self.config)} Da")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = get_format_extension(output_format)
        fig_filename = f"complete_enumeration_{limits['n_points']}_{timestamp}{ext}"
        fig_path = os.path.join(self.output_dir, fig_filename) if save_fig else None

        self._print(f"\n  Generating figure...")

        plot_data = self.precomputed_df[['delta_d', 'ci_lp', 'ci_up', 'ci_lp_up_abs_max']].copy()

        fig, ax, plot_stats = self.plot_hageman_figure(
            data=plot_data,
            eac=limits['eac'],
            delta_d_limit=limits['delta_d_limit'],
            panel_label=panel_label,
            figsize=figsize,
            save_path=fig_path,
            show_plot=show_plot,
            output_format=output_format
        )

        limits['figure_path'] = fig_path

        self._print(f"{'='*70}\n")

        return limits


# ==============================================================================
# CONVENIENCE FUNCTION
# ==============================================================================

def run_hageman_resampling(context: 'ExperimentContext',
                           config: Dict[str, Any],
                           precomputed_df: pd.DataFrame,
                           data_structure: Dict[str, Any],
                           mode: str = None,
                           n_simulations: int = None,
                           n_rounds: int = None,
                           panel_label: str = None,
                           verbose: bool = None,
                           output_format: str = None) -> Dict[str, Any]:
    """
    Convenience function to run resampling from config.

    Parameters
    ----------
    context : ExperimentContext
        Experiment context with parameters
    config : dict
        Configuration dictionary
    precomputed_df : pd.DataFrame
        Precomputed statistics
    data_structure : dict
        Data structure from data_loader
    mode : str, optional
        Override mode from config
    n_simulations : int, optional
        Override n_simulations from config
    n_rounds : int, optional
        Override n_rounds from config
    panel_label : str, optional
        Panel label for figure
    verbose : bool, optional
        Print progress (default from context)
    output_format : str, optional
        Figure format: 'tiff' (default), 'png', or 'pdf'.
        If None, reads from config.

    Returns
    -------
    dict
        Resampling results
    """
    if mode is None:
        mode = get_config_value(config, 'ANALYSIS_OPTIONS', 'resampling_mode', 'B')
    if n_simulations is None:
        n_simulations = get_config_int(config, 'ANALYSIS_OPTIONS', 'resampling_n_simulations', 5)
    if n_rounds is None:
        n_rounds = get_config_int(config, 'ANALYSIS_OPTIONS', 'resampling_n_rounds', 1)
    if verbose is None:
        verbose = context.verbose
    if output_format is None:
        output_format = get_config_value(config, 'OUTPUT_SETTINGS', 'figure_format', DEFAULT_FORMAT)

    save_figures = get_config_bool(config, 'OUTPUT_SETTINGS', 'save_plots', True)
    generate_plots = get_config_bool(config, 'OUTPUT_SETTINGS', 'generate_hageman_plots', True)

    resampler = HagemanResampler(
        context=context,
        config=config,
        precomputed_df=precomputed_df,
        data_structure=data_structure,
        verbose=verbose
    )

    results = resampler.run_resampling(
        mode=mode,
        n_simulations=n_simulations,
        n_rounds=n_rounds,
        generate_plot=generate_plots,
        save_fig=save_figures,
        panel_label=panel_label,
        show_plot=False,  # Don't block in automated runs
        output_format=output_format
    )

    return results


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    print("RESAMPLING MODULE - TEST")
    print("Run main.py to execute the full pipeline.")
