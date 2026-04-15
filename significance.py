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
HDX-MS HYBRID SIGNIFICANCE TESTING MODULE
================================================================================

PURPOSE:
    Implements the hybrid NHST method for complementary analysis

    Reference:
      Hageman, T. S.; Weis, D. D. Reliable Identification of Significant
      Differences in Differential Hydrogen Exchange-Mass Spectrometry
      Measurements Using a Hybrid Significance Testing Approach.
      Anal. Chem. 2019, 91 (13), 8008-8016.
      https://doi.org/10.1021/acs.analchem.9b01325

OUTPUTS:
    - Per-row hybrid significance results
    - Volcano plot across labeling times
    - CSV export of detailed significance results

DEPENDENCIES:
    - numpy, pandas, matplotlib, scipy
    - config_parser.py
    - statistics_util.py
    - plotting.py

================================================================================
"""

import os
from collections import Counter
from typing import Dict, List, Optional, Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.ticker import (
    LogFormatterMathtext,
    LogLocator,
    NullFormatter,
    NullLocator
)

from config_parser import get_config_value
from statistics_util import (
    calculate_satterthwaite_df,
    calculate_standard_error,
    get_cached_t_value
)
from plotting import (
    create_figure,
    save_figure,
    style_axes_publication,
    get_figure_settings,
    DEFAULT_DPI
)

if TYPE_CHECKING:
    from experiment import ExperimentContext


class HybridSignificanceTester:
    _MIN_PLOT_PVALUE = np.finfo(float).tiny

    def __init__(self,
                 context: 'ExperimentContext',
                 config: Dict[str, Any],
                 verbose: bool = None):
        """
        Parameters
        ----------
        context : ExperimentContext
        config : dict
        verbose : bool, optional
        """
        self.context = context
        self.config = config
        self.verbose = verbose if verbose is not None else context.verbose
        self.output_dir = get_config_value(
            config, 'OUTPUT_SETTINGS', 'output_directory', './outputs'
        )

    def _ensure_output_dir(self) -> None:
        """Create output directory if it does not exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _get_optional_figure_limit(self, key: str) -> Optional[float]:
        """Read an optional numeric figure limit from config."""
        value = get_config_value(self.config, 'FIGURE_SETTINGS', key, '')
        if value is None:
            return None
        value_str = str(value).strip()
        if not value_str:
            return None
        return float(value_str)

    def calculate_pooled_sd(self,
                            null_data: pd.DataFrame,
                            uptake_lookup: Dict,
                            n_draws: int = 5) -> float:
        """
        Compute pooled standard deviation 

        Parameters
        ----------
        null_data : pd.DataFrame
            Null experiment data with individual replicate rows
        uptake_lookup : dict
            Mapping of (Sequence, Replicate number, HX time) to
            Uptake (Da), as created by DataLoader.create_uptake_lookup()
        n_draws : int, optional
            Number of random non-overlapping group selections
            (default 5, following Hageman & Weis 2019)

        Returns
        -------
        float
            Pooled standard deviation (s_p)
        """
        replicates = list(self.context.replicates)
        group_size = self.context.group_size

        peptide_times = (
            null_data.groupby(['Sequence', 'HX time'])
            .size()
            .index.tolist()
        )

        all_sds = []

        for _ in range(n_draws):
            shuffled = list(np.random.permutation(replicates))
            ref_reps = shuffled[:group_size]
            sample_reps = shuffled[group_size:2 * group_size]

            for seq, time in peptide_times:
                ref_uptakes = []
                for rep in ref_reps:
                    key = (seq, rep, time)
                    if key in uptake_lookup:
                        ref_uptakes.append(uptake_lookup[key])

                sample_uptakes = []
                for rep in sample_reps:
                    key = (seq, rep, time)
                    if key in uptake_lookup:
                        sample_uptakes.append(uptake_lookup[key])

                if len(ref_uptakes) >= 2:
                    all_sds.append(np.std(ref_uptakes, ddof=1))
                if len(sample_uptakes) >= 2:
                    all_sds.append(np.std(sample_uptakes, ddof=1))

        all_sds = np.array(all_sds)
        pooled_sd = float(np.sqrt(np.mean(all_sds ** 2)))

        if self.verbose:
            print(f"    Pooled SD from {len(all_sds)} standard"
                  f" deviations ({n_draws} random draws of"
                  f" 2x{group_size} replicates)")

        return pooled_sd

    def calculate_row_threshold(self,
                                s_p: float,
                                n_ref: int,
                                n_cand: int,
                                alpha: float) -> float:
        """
        Compute the delta_HX significance threshold for a specific
        (n_ref, n_cand) pair.
        """
        df = n_ref + n_cand - 2
        k = get_cached_t_value(df, alpha)
        threshold = k * np.sqrt(s_p ** 2 / n_ref + s_p ** 2 / n_cand)
        return float(threshold)

    def evaluate_significance(self,
                              reference_df: pd.DataFrame,
                              candidate_df: pd.DataFrame,
                              s_p: float,
                              alpha: float) -> Dict[str, Any]:
        """
        Evaluate significance for all peptide-time combinations.

        Assumes (Sequence, HX time) keys are unique in both DataFrames.
        This invariant is enforced at load time by validate_aggregated_data().
        Rows are only evaluated when both sides have at least the expected
        replicate count from context.group_size.
        """
        ref_lookup = {
            (row['Sequence'], row['HX time']): row
            for _, row in reference_df.iterrows()
        }
        cand_lookup = {
            (row['Sequence'], row['HX time']): row
            for _, row in candidate_df.iterrows()
        }

        ref_keys = set(ref_lookup.keys())
        cand_keys = set(cand_lookup.keys())

        unmatched_cand_keys = cand_keys - ref_keys
        unmatched_ref_keys = ref_keys - cand_keys
        n_unmatched_cand = len(unmatched_cand_keys)
        n_unmatched_ref = len(unmatched_ref_keys)

        if self.verbose and n_unmatched_cand > 0:
            print(f"    WARNING: {n_unmatched_cand} candidate"
                  f" peptide-time combinations have no matching"
                  f" reference row")
        if self.verbose and n_unmatched_ref > 0:
            print(f"    WARNING: {n_unmatched_ref} reference"
                  f" peptide-time combinations have no matching"
                  f" candidate row")

        results = []
        n_skipped_low_n = 0
        expected_n = self.context.group_size

        for key, cand_row in cand_lookup.items():
            if key in unmatched_cand_keys:
                continue

            ref_row = ref_lookup[key]
            seq, time = key

            uptake_ref = ref_row['Uptake (Da)']
            uptake_cand = cand_row['Uptake (Da)']
            sd_ref = ref_row['Uptake SD (Da)']
            sd_cand = cand_row['Uptake SD (Da)']
            n_ref = int(ref_row['Replicate number'])
            n_cand = int(cand_row['Replicate number'])
            ref_protein_state = ref_row.get(
                'Protein state', ref_row.get('Protein State', 'Reference')
            )
            cand_protein_state = cand_row.get(
                'Protein state', cand_row.get('Protein State', 'Candidate')
            )

            if n_ref < expected_n or n_cand < expected_n:
                n_skipped_low_n += 1
                continue

            delta_hx = uptake_cand - uptake_ref
            abs_delta_hx = abs(delta_hx)
            row_threshold = self.calculate_row_threshold(
                s_p, n_ref, n_cand, alpha
            )

            se = calculate_standard_error(sd_ref, sd_cand, n_ref, n_cand)
            df = calculate_satterthwaite_df(sd_ref, sd_cand, n_ref, n_cand)

            if se > 0 and df > 0:
                t_stat = abs_delta_hx / se
                p_value = float(2.0 * stats.t.sf(t_stat, df))
            else:
                p_value = 1.0 if abs_delta_hx == 0 else 0.0

            neg_log10_p = float(-np.log10(p_value)) if p_value > 0 else 300.0

            pass_threshold = abs_delta_hx > row_threshold
            pass_pvalue = p_value < alpha
            is_significant = pass_threshold and pass_pvalue

            results.append({
                'Sequence': seq,
                'HX time': time,
                'delta_hx': delta_hx,
                'abs_delta_hx': abs_delta_hx,
                'threshold': row_threshold,
                'p_value': p_value,
                'neg_log10_p': neg_log10_p,
                'sd_ref': sd_ref,
                'sd_cand': sd_cand,
                'n_ref': n_ref,
                'n_cand': n_cand,
                'uptake_ref': uptake_ref,
                'uptake_cand': uptake_cand,
                'ref_protein_state': ref_protein_state,
                'cand_protein_state': cand_protein_state,
                'pass_threshold': pass_threshold,
                'pass_pvalue': pass_pvalue,
                'significant': is_significant
            })

        if self.verbose and n_skipped_low_n > 0:
            print(f"    WARNING: {n_skipped_low_n} rows skipped"
                  f" because replicate count was below the expected"
                  f" group_size={expected_n}")

        sig_list = [r for r in results if r['significant']]

        if results:
            n_pairs = [(r['n_ref'], r['n_cand']) for r in results]
            most_common_pair = Counter(n_pairs).most_common(1)[0][0]
            rep_threshold = self.calculate_row_threshold(
                s_p, most_common_pair[0], most_common_pair[1], alpha
            )
        else:
            rep_threshold = 0.0

        return {
            'n_total': len(results),
            'n_significant': len(sig_list),
            'n_not_significant': len(results) - len(sig_list),
            'n_skipped_low_n': n_skipped_low_n,
            'n_unmatched_cand': n_unmatched_cand,
            'n_unmatched_ref': n_unmatched_ref,
            'expected_replicates': expected_n,
            'alpha': alpha,
            's_p': s_p,
            'representative_threshold': rep_threshold,
            'significant_peptides': sig_list,
            'all_results': results
        }

    def generate_volcano_plot(self,
                              results: Dict[str, Any],
                              output_format: str = None,
                              save_fig: bool = True,
                              show_plot: bool = False,
                              display_name: str = '',
                              candidate_name: str = '') -> Optional[str]:
        """
        Generate a combined volcano plot across all labeling times.
        """
        all_results = results['all_results']
        threshold = results['representative_threshold']
        alpha = results['alpha']

        if not all_results:
            return None

        self._ensure_output_dir()

        fig_settings = get_figure_settings(self.config, 'volcano')
        figsize = fig_settings.get('figsize', (3.7, 4.0))
        dpi = fig_settings.get('dpi', DEFAULT_DPI)
        axis_fontsize = fig_settings.get('axis_title_fontsize', 8)
        tick_fontsize = fig_settings.get('axis_number_fontsize', 7)
        dot_size = fig_settings.get('dot_size', 8)

        fig, ax = create_figure(figsize=figsize, figure_type='volcano')

        non_sig = [r for r in all_results if not r['significant']]
        sig = [r for r in all_results if r['significant']]

        if non_sig:
            x_ns = [r['delta_hx'] for r in non_sig]
            y_ns = [
                max(r['p_value'], self._MIN_PLOT_PVALUE)
                for r in non_sig
            ]
            ax.scatter(x_ns, y_ns, c="#040404", s=dot_size,
                       alpha=1, edgecolors='none', zorder=2,
                       label='Not significant')

        if sig:
            x_s = [r['delta_hx'] for r in sig]
            y_s = [
                max(r['p_value'], self._MIN_PLOT_PVALUE)
                for r in sig
            ]
            ax.scatter(x_s, y_s, c="#f40606", s=dot_size * 1.8,
                       alpha=1, edgecolors='none', zorder=3,
                       label='Significant')

        x_min = self._get_optional_figure_limit('volcano_xmin')
        x_max = self._get_optional_figure_limit('volcano_xmax')
        y_min = self._get_optional_figure_limit('volcano_ymin')
        y_max = self._get_optional_figure_limit('volcano_ymax')
        auxiliary_line = self._get_optional_figure_limit(
            'volcano_auxiliary_vertical_line'
        )
        if x_min is None:
            x_min = -0.7
        if x_max is None:
            x_max = 0.7
        if y_min is None:
            y_min = min(
                max(r['p_value'], self._MIN_PLOT_PVALUE)
                for r in all_results
            )
        if y_max is None:
            y_max = 10.0
        ax.set_xlim(left=x_min, right=x_max)
        ax.set_yscale('log')
        ax.set_ylim(bottom=y_min, top=y_max)
        ax.invert_yaxis()

        y_threshold = max(alpha, self._MIN_PLOT_PVALUE)
        y_top, y_bottom = ax.get_ylim()

        ax.vlines(x=threshold, ymin=y_threshold, ymax=y_bottom, color='red',
                  linestyle='--', linewidth=0.65, zorder=1)
        ax.vlines(x=-threshold, ymin=y_threshold, ymax=y_bottom, color='red',
                  linestyle='--', linewidth=0.65, zorder=1)
        ax.hlines(y=y_threshold, xmin=x_min, xmax=-threshold, color='red',
                  linestyle='--', linewidth=0.65, zorder=1)
        ax.hlines(y=y_threshold, xmin=threshold, xmax=x_max, color='red',
                  linestyle='--', linewidth=0.65, zorder=1)
        if auxiliary_line is not None:
            ax.vlines(x=auxiliary_line, ymin=y_top, ymax=y_bottom,
                      color='black', linestyle='--', linewidth=0.65, zorder=1)
            ax.vlines(x=-auxiliary_line, ymin=y_top, ymax=y_bottom,
                      color='black', linestyle='--', linewidth=0.65, zorder=1)

        ax.set_xlabel(
            r'$\Delta \overline{\mathrm{HX}}$ (Da)',
            fontsize=axis_fontsize
        )
        ax.set_ylabel(r'$p$-value', fontsize=axis_fontsize)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())

        style_axes_publication(ax)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.tick_params(axis='y', which='minor', length=0)
        fig.subplots_adjust(left=0.22, bottom=0.14, top=0.98)

        figure_path = None
        if save_fig:
            fmt = output_format or get_config_value(
                self.config, 'OUTPUT_SETTINGS', 'figure_format', 'tiff'
            )
            safe_display = display_name.replace(' ', '_') if display_name else 'protein'
            safe_cand = candidate_name.replace(' ', '_') if candidate_name else 'candidate'
            filename = f"volcano_{safe_display}_{safe_cand}"
            output_path = os.path.join(self.output_dir, filename)
            figure_path = save_figure(
                fig, output_path, output_format=fmt, dpi=dpi
            )

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return figure_path

    def save_results_csv(self,
                         results: Dict[str, Any],
                         display_name: str = '',
                         candidate_name: str = '') -> Optional[str]:
        """
        Save detailed per-peptide results to CSV.
        """
        if not results['all_results']:
            return None

        self._ensure_output_dir()

        df = pd.DataFrame(results['all_results'])
        safe_display = display_name.replace(' ', '_') if display_name else 'protein'
        safe_cand = candidate_name.replace(' ', '_') if candidate_name else 'candidate'
        filename = f"significance_{safe_display}_{safe_cand}.csv"
        csv_path = os.path.join(self.output_dir, filename)
        df.to_csv(csv_path, index=False)
        return csv_path
