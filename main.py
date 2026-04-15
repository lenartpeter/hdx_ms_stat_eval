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
HDX-MS EQUIVALENCE TESTING - MAIN ORCHESTRATION MODULE
================================================================================

PURPOSE:
    Main entry point for the HDX-MS TOST equivalence testing pipeline.
    Reads configuration from config.txt and orchestrates execution of all
    analysis modules based on specified settings.

PIPELINE OVERVIEW:
    1. Parse configuration, validate, create ExperimentContext
    2. For each unique null experiment:
       - Load data, set context data structure
       - Precompute 314k statistics (or load existing)
       - Calculate limits using selected methods
    3. For each protein section:
       - Use precomputed data from its null experiment
       - Evaluate candidates against selected limits

INPUTS:
    - config.txt configuration file
    - Null experiment CSV files
    - Reference and candidate CSV files

OUTPUTS:
    - Console output with analysis results
    - Precomputed statistics files
    - Evaluation results

DEPENDENCIES:
    - config_parser (configuration parsing and validation)
    - experiment (ExperimentContext)
    - statistics (statistical calculations)
    - data_loader (data loading and validation)
    - precompute (statistics precomputation)
    - Analysis modules: resampling, monte_carlo, direct_percentile, partitioned_limits

USAGE:
    python main.py --config config.txt

================================================================================
"""

import os
import sys
import argparse
import numbers
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

# 
from config_parser import (
    parse_config,
    validate_config,
    get_config_value,
    get_config_bool,
    get_config_float,
    get_config_int,
    get_config_list,
    get_protein_sections,
    get_candidates_from_section,
    ConfigError
)

from statistics_util import (
    calculate_satterthwaite_df,
    calculate_standard_error,
    calculate_welch_ci
)

from data_loader import (
    DataLoader,
    calculate_peptide_length,
    validate_null_experiment_data
)

from experiment import ExperimentContext
from precompute import Precomputer, build_max_arrays
from report_generator import OutputCapture, generate_pdf_report
from report_formatting import format_report_limit
from plotting import plot_complete_enumeration_figure

# Optional imports - handled gracefully if not available
try:
    from resampling import HagemanResampler
    RESAMPLING_AVAILABLE = True
except ImportError:
    RESAMPLING_AVAILABLE = False

try:
    from significance import HybridSignificanceTester
    SIGNIFICANCE_AVAILABLE = True
except ImportError:
    SIGNIFICANCE_AVAILABLE = False


# ==============================================================================
# OUTPUT FORMATTING
# ==============================================================================

LINE_WIDTH = 60

def print_section_header(text: str) -> None:
    """Print a major section header with = lines."""
    print("=" * LINE_WIDTH)
    print(text)
    print("=" * LINE_WIDTH)


def print_method_header(text: str) -> None:
    """Print a method section header with blank line before for separation."""
    print()  # Blank line before new method section
    print("=" * LINE_WIDTH)
    print(text)
    print("-" * LINE_WIDTH)


def print_separator() -> None:
    """Print a separator line."""
    print("=" * LINE_WIDTH)


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

def build_candidate_results(reference_df, candidate_df,
                            alpha_ci: float = 0.10) -> List[Dict[str, Any]]:
    """Build per-measurement comparison statistics for a candidate sample."""
    all_results = []

    for _, cand_row in candidate_df.iterrows():
        seq = cand_row['Sequence']
        time = cand_row['HX time']

        ref_match = reference_df[
            (reference_df['Sequence'] == seq) &
            (reference_df['HX time'] == time)
        ]

        if len(ref_match) == 0:
            continue

        ref_row = ref_match.iloc[0]

        uptake_ref = ref_row['Uptake (Da)']
        uptake_cand = cand_row['Uptake (Da)']
        sd_ref = ref_row['Uptake SD (Da)']
        sd_cand = cand_row['Uptake SD (Da)']
        n_ref = int(ref_row['Replicate number'])
        n_cand = int(cand_row['Replicate number'])

        # Get protein state names
        ref_protein_state = ref_row.get('Protein state', ref_row.get('Protein State', 'Reference'))
        cand_protein_state = cand_row.get('Protein state', cand_row.get('Protein State', 'Candidate'))

        # Calculate statistics using imported functions
        delta_d = uptake_cand - uptake_ref
        se = calculate_standard_error(sd_ref, sd_cand, n_ref, n_cand)
        df = calculate_satterthwaite_df(sd_ref, sd_cand, n_ref, n_cand)

        # Use scipy for t-critical (imported at module level would add dependency)
        from scipy import stats
        t_crit = stats.t.ppf(1 - alpha_ci / 2, df)
        margin = t_crit * se

        ci_lp = delta_d - margin
        ci_up = delta_d + margin
        ci_abs_max = max(abs(ci_lp), abs(ci_up))

        result = {
            'Sequence': seq,
            'HX time': time,
            'delta_d': delta_d,
            'delta_d_abs': abs(delta_d),
            'ci_lp': ci_lp,
            'ci_up': ci_up,
            'ci_abs_max': ci_abs_max,
            'se': se,
            'uptake_ref': uptake_ref,
            'sd_ref': sd_ref,
            'uptake_cand': uptake_cand,
            'sd_cand': sd_cand,
            'ref_protein_state': ref_protein_state,
            'cand_protein_state': cand_protein_state
        }

        all_results.append(result)

    return all_results


def evaluate_candidate(reference_df, candidate_df, limits: Dict[str, float],
                       alpha_ci: float = 0.10) -> Dict[str, Any]:
    """
    Evaluate a biosimilar candidate against established limits.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Reference protein data (aggregated format)
    candidate_df : pd.DataFrame
        Candidate protein data (aggregated format)
    limits : dict
        Dictionary with 'delta_d_limit' and 'eac' keys
    alpha_ci : float, optional
        Significance level for CI (default 0.10)

    Returns
    -------
    dict
        Dictionary with:
        - n_total: Total peptide-time combinations evaluated
        - n_pass: Number passing both criteria
        - n_fail: Number failing at least one criterion
        - passed: True if all pass
        - failed_peptides: List of failed peptide details
        - all_results: List of all evaluation results
    """
    delta_d_limit = limits['delta_d_limit']
    eac_limit = limits['eac']

    failed_peptides = []
    failed_delta_d = 0
    failed_eac = 0
    failed_both = 0
    all_results = build_candidate_results(reference_df, candidate_df, alpha_ci)

    for result in all_results:
        pass_delta_d = result['delta_d_abs'] < delta_d_limit
        pass_ci = result['ci_abs_max'] < eac_limit
        pass_both = pass_delta_d and pass_ci

        result['pass_delta_d'] = pass_delta_d
        result['pass_ci'] = pass_ci
        result['pass_both'] = pass_both

        if not pass_both:
            failed_peptides.append(result)
            if not pass_delta_d:
                failed_delta_d += 1
            if not pass_ci:
                failed_eac += 1
            if not pass_delta_d and not pass_ci:
                failed_both += 1

    n_total = len(all_results)
    n_pass = sum(1 for r in all_results if r['pass_both'])
    n_fail = n_total - n_pass

    return {
        'n_total': n_total,
        'n_pass': n_pass,
        'n_fail': n_fail,
        'failed_delta_d': failed_delta_d,
        'failed_eac': failed_eac,
        'failed_both': failed_both,
        'passed': n_fail == 0,
        'failed_peptides': failed_peptides,
        'all_results': all_results
    }


def print_measurement_uptake_details(result: Dict) -> None:
    """Print uptake summary fields shared by evaluation outputs."""
    ref_name = result.get('ref_protein_state', 'Reference')
    cand_name = result.get('cand_protein_state', 'Candidate')

    if isinstance(result.get('uptake_ref'), numbers.Real):
        print(f"      Av_Uptake_{ref_name}: {result['uptake_ref']:.4f} Da")
    if isinstance(result.get('sd_ref'), numbers.Real):
        print(f"      Uptake_SD_{ref_name}: {result['sd_ref']:.4f} Da")
    if isinstance(result.get('uptake_cand'), numbers.Real):
        print(f"      Av_Uptake_{cand_name}: {result['uptake_cand']:.4f} Da")
    if isinstance(result.get('sd_cand'), numbers.Real):
        print(f"      Uptake_SD_{cand_name}: {result['sd_cand']:.4f} Da")


def print_failed_peptide_details(fp: Dict) -> None:
    """Print detailed information for a failed equivalence measurement."""
    print(f"      Peptide sequence: {fp['Sequence']}")
    print(f"      Labeling time: {fp['HX time']}s")
    print(f"      |dD|: {fp.get('delta_d_abs', abs(fp['delta_d'])):.4f} Da")
    print(f"      |CI_LP,UP|_max: {fp['ci_abs_max']:.4f} Da")
    print_measurement_uptake_details(fp)
    print()  # Blank line between failed measurements


def print_significance_measurement_details(result: Dict) -> None:
    """Print detailed information for a significant measurement."""
    print(f"      Peptide sequence: {result['Sequence']}")
    print(f"      Labeling time: {result['HX time']}s")
    print(f"      dHX: {result['delta_hx']:.4f} Da")
    print(f"      p value: {result['p_value']:.2e}")
    print_measurement_uptake_details(result)
    print()  # Blank line between failed measurements


def format_failure_breakdown(eval_result: Dict[str, Any]) -> str:
    """Format the failure-category breakdown."""
    return (
        f"{eval_result['n_fail']}/{eval_result['n_total']} peptide-labeling time combinations "
        f"({eval_result['failed_delta_d']} failing |ΔD|_limit, "
        f"{eval_result['failed_eac']} failing EAC, "
        f"{eval_result['failed_both']} failing both)"
    )


def print_evaluation_result(cand_name: str, cand_filename: str,
                            eval_result: Dict, limits: Dict[str, float] = None,
                            method_name: str = "",
                            config: Dict[str, Any] = None) -> None:
    """Print evaluation result for a candidate."""
    prefix = f"  {cand_name} ({cand_filename})"
    if method_name:
        prefix = f"    [{method_name}]"

    # Print limits if provided
    if limits:
        print(
            f"{prefix} - Limits: |dD|_limit = {format_report_limit(limits['delta_d_limit'], config or {})} Da, "
            f"EAC = {format_report_limit(limits['eac'], config or {})} Da"
        )

    if eval_result['passed']:
        print(f"{prefix}: PASSED ({eval_result['n_pass']}/{eval_result['n_total']} peptide-labeling time combinations)")
    else:
        print(f"{prefix}: FAILED ({format_failure_breakdown(eval_result)})")
        print("      Failed measurements:")
        for fp in eval_result['failed_peptides']:
            print_failed_peptide_details(fp)


# ==============================================================================
# NULL EXPERIMENT PROCESSING
# ==============================================================================

def process_null_experiment(null_exp_path: str, context: 'ExperimentContext',
                            config: Dict, loader: DataLoader,
                            precomputed_cache: Dict) -> Dict[str, Any]:
    """
    Process a null experiment file: load data, precompute statistics, calculate limits.

    Parameters
    ----------
    null_exp_path : str
        Path to null experiment CSV file
    context : ExperimentContext
        Experiment context with all parameters
    config : dict
        Configuration dictionary
    loader : DataLoader
        Data loader instance
    precomputed_cache : dict
        Cache for precomputed results (by null_exp_path)

    Returns
    -------
    dict
        Dictionary with structure, precomputed_df, and limits
    """
    # Check cache first
    if null_exp_path in precomputed_cache:
        return precomputed_cache[null_exp_path]

    null_exp_folder = os.path.basename(os.path.dirname(null_exp_path))

    # --------------------------------------------------------------------------
    # II. Load null experiment data
    # --------------------------------------------------------------------------
    print()  # Blank line before section
    print("II. Null experiment data:")

    null_data, structure = loader.load_null_experiment(null_exp_path)

    # Validate data against expected configuration
    validate_null_experiment_data(null_data, context.n_replicates)

    # Update context with data structure
    context.set_data_structure(
        time_points=structure['time_points'],
        peptides=structure['peptides']
    )

    print(f"{null_exp_folder} - Rows: {structure['n_rows']:,}  Peptides: {structure['n_peptides']}  "
          f"Replicates: {structure['n_replicates']}")
    print(f"Labeling times: {structure['n_times']} - {structure['time_points']}")

    # Validate context
    context.validate()

    # --------------------------------------------------------------------------
    # III. Create lookup and pairings
    # --------------------------------------------------------------------------
    print()  # Blank line before section
    print("III. Creating pairings and lookup table")

    uptake_lookup = loader.create_uptake_lookup(null_data)
    pairings = loader.generate_all_pairings(context.replicates, context.group_size)

    print(f"Group size: {context.group_size}  Non-overlapping replicate pairings: {len(pairings)}")

    # --------------------------------------------------------------------------
    # Precomputation
    # --------------------------------------------------------------------------
    precomputer = Precomputer(context, config, verbose=context.verbose)

    precomputed_df = precomputer.get_or_create_precomputed_data(
        raw_data=null_data,
        uptake_lookup=uptake_lookup,
        pairings=pairings,
        data_structure=structure,
        null_experiment_file=null_exp_path
    )

    # --------------------------------------------------------------------------
    # Generate uptake range figure (if enabled)
    # --------------------------------------------------------------------------
    generate_uptake_range = get_config_bool(config, 'OUTPUT_SETTINGS', 'generate_uptake_range_figure', True)
    if generate_uptake_range:
        output_format = get_config_value(config, 'OUTPUT_SETTINGS', 'figure_format', 'tiff')
        uptake_range_path = precomputer.generate_uptake_range_figure(
            raw_df=null_data,
            output_format=output_format,
            show_plot=False
        )
        if uptake_range_path:
            print(f"  Uptake range figure saved: {uptake_range_path}")

    # --------------------------------------------------------------------------
    # Calculate limits using all selected methods
    # --------------------------------------------------------------------------
    results = {
        'structure': structure,
        'precomputed_df': precomputed_df,
        'limits': {},
        'context': context
    }

    # Complete Enumeration
    if get_config_bool(config, 'ANALYSIS_OPTIONS', 'run_complete_enumeration', True):
        print()  # Blank line before section
        print("=" * LINE_WIDTH)
        print("COMPLETE ENUMERATION")
        print("-" * LINE_WIDTH)
        global_limits = precomputer.get_global_limits(precomputed_df)
        print(f"Total values examined: {len(precomputed_df):,}")
        print()  # Blank line before Global limits:
        print("Global limits:")
        print(
            f"EAC = {format_report_limit(global_limits['eac'], config)} Da    "
            f"|ΔD|_limit = {format_report_limit(global_limits['delta_d_limit'], config)} Da"
        )

        # Save complete enumeration data if config option is set
        save_ce_data = get_config_bool(config, 'OUTPUT_SETTINGS', 'save_complete_enumeration_data', False)
        if save_ce_data:
            output_dir = get_config_value(config, 'OUTPUT_SETTINGS', 'output_directory', './outputs')
            ce_data_path = os.path.join(output_dir, f"complete_enumeration_data_{len(precomputed_df)}.csv")
            precomputed_df.to_csv(ce_data_path, index=False)
            print(f"Complete enumeration data saved: {ce_data_path}")

        # Generate complete enumeration figure if config option is set
        generate_ce_plot = get_config_bool(config, 'OUTPUT_SETTINGS', 'generate_complete_enumeration_plot', False)
        if generate_ce_plot:
            output_dir = get_config_value(config, 'OUTPUT_SETTINGS', 'output_directory', './outputs')
            save_plots = get_config_bool(config, 'OUTPUT_SETTINGS', 'save_plots', True)
            figure_format = get_config_value(config, 'OUTPUT_SETTINGS', 'figure_format', 'tiff')

            ce_fig_result = plot_complete_enumeration_figure(
                precomputed_df=precomputed_df,
                config=config,
                output_dir=output_dir,
                output_format=figure_format,
                save_fig=save_plots,
                show_plot=False,
                verbose=False
            )
            if ce_fig_result.get('figure_path'):
                print(f"Complete enumeration figure saved: {ce_fig_result['figure_path']}")

        print("=" * LINE_WIDTH)

        results['limits']['complete_enumeration'] = {
            'delta_d_limit': global_limits['delta_d_limit'],
            'eac': global_limits['eac']
        }

    # Hageman Resampling
    if get_config_bool(config, 'ANALYSIS_OPTIONS', 'run_resampling', False):
        print()  # Blank line before section
        print("=" * LINE_WIDTH)
        print("RESAMPLING method by Hageman et al.")
        print("[https://doi.org/10.1021/acs.analchem.0c05279]")
        print("-" * LINE_WIDTH)

        if not RESAMPLING_AVAILABLE:
            print("SKIPPED: resampling.py not available")
        else:
            try:
                mode = get_config_value(config, 'ANALYSIS_OPTIONS', 'resampling_mode', 'B')
                n_simulations = get_config_int(config, 'ANALYSIS_OPTIONS', 'resampling_n_simulations', 5)
                n_rounds = get_config_int(config, 'ANALYSIS_OPTIONS', 'resampling_n_rounds', 1)
                generate_plots = get_config_bool(config, 'OUTPUT_SETTINGS', 'generate_hageman_plots', True)
                save_plots = get_config_bool(config, 'OUTPUT_SETTINGS', 'save_plots', True)
                save_resampling_results = get_config_bool(config, 'OUTPUT_SETTINGS', 'save_resampling_results', True)
                figure_format = get_config_value(config, 'OUTPUT_SETTINGS', 'figure_format', 'tiff')

                resampler = HagemanResampler(
                    context=context,
                    config=config,
                    precomputed_df=precomputed_df,
                    data_structure=structure,
                    verbose=False  # We'll print our own formatted output
                )

                resampling_results = resampler.run_resampling(
                    mode=mode,
                    n_simulations=n_simulations,
                    n_rounds=n_rounds,
                    generate_plot=generate_plots,
                    save_fig=save_plots,
                    save_results=save_resampling_results,
                    show_plot=False,
                    output_format=figure_format
                )

                data_points_per_sim = structure['n_peptides'] * structure['n_times']
                total_data_points = data_points_per_sim * n_simulations
                print(f"No. of simulations: {n_simulations}, Data points per simulation: {data_points_per_sim:,}")
                print(f"Total data points: {total_data_points:,}, Mode: {mode}")
                print()  # Blank line before Limits
                print("Limits:")
                print(
                    f"EAC = {format_report_limit(resampling_results['eac_max'], config)} Da    "
                    f"|ΔD|_limit = {format_report_limit(resampling_results['delta_d_limit_max'], config)} Da"
                )

                if resampling_results.get('figure_paths'):
                    for fig_path in resampling_results['figure_paths']:
                        if fig_path:
                            print(f"Figure saved: {fig_path}")

                results['limits']['resampling'] = {
                    'delta_d_limit': resampling_results['delta_d_limit_max'],
                    'eac': resampling_results['eac_max']
                }
                results['resampling_full'] = resampling_results

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

        print("=" * LINE_WIDTH)

    # Monte Carlo
    if get_config_bool(config, 'ANALYSIS_OPTIONS', 'run_monte_carlo', True):
        print()  # Blank line before section
        print("=" * LINE_WIDTH)
        print("MONTE CARLO Method")
        print("-" * LINE_WIDTH)

        try:
            from monte_carlo import MonteCarloSimulator

            simulator = MonteCarloSimulator(
                context=context,
                precomputed_df=precomputed_df,
                time_points=structure['time_points'],
                config=config,
                verbose=context.verbose
            )

            mc_results = simulator.run_simulation(
                n_simulations=context.n_simulations,
                batch_size=context.batch_size,
                random_seed=context.random_seed,
                dD_limit_abs=context.dD,
                EAC=context.EAC
            )

            results['limits']['monte_carlo'] = {
                'delta_d_limit': mc_results['delta_d_limit'],
                'eac': mc_results['eac_limit']
            }
            results['monte_carlo_full'] = mc_results

        except ImportError:
            print("SKIPPED: monte_carlo.py not available")
        except Exception as e:
            print(f"ERROR: {e}")

        print("=" * LINE_WIDTH)

    # Direct Percentile
    if get_config_bool(config, 'ANALYSIS_OPTIONS', 'run_direct_percentile', True):
        print()  # Blank line before section
        print("=" * LINE_WIDTH)
        print("DIRECT PERCENTILE Method")
        print("-" * LINE_WIDTH)

        try:
            from direct_percentile import DirectPercentileAnalyzer

            analyzer = DirectPercentileAnalyzer(
                context=context,
                precomputed_df=precomputed_df,
                time_points=structure['time_points'],
                config=config,
                verbose=context.verbose
            )

            dp_results = analyzer.calculate_acceptance_limits()

            # Print figure paths (centralized in main.py)
            if dp_results.get('figure_paths'):
                for fig_path in dp_results['figure_paths']:
                    if fig_path:
                        print(f"Figure saved: {fig_path}")

            results['limits']['direct_percentile'] = {
                'delta_d_limit': dp_results['delta_d_limit'],
                'eac': dp_results['eac']
            }
            results['direct_percentile_full'] = dp_results

        except ImportError:
            print("SKIPPED: direct_percentile.py not available")
        except Exception as e:
            print(f"ERROR: {e}")

        print("=" * LINE_WIDTH)

    # Partitioned Limits
    if get_config_bool(config, 'ANALYSIS_OPTIONS', 'run_partitioned_limits', False):
        print()  # Blank line before section
        print("=" * LINE_WIDTH)
        print("PARTITIONED LIMITS Method")
        print("-" * LINE_WIDTH)

        try:
            from partitioned_limits import PartitionedLimitsAnalyzer

            pl_analyzer = PartitionedLimitsAnalyzer(
                context=context,
                precomputed_df=precomputed_df,
                config=config,
                verbose=context.verbose
            )

            results['partitioned_analyzer'] = pl_analyzer
            print("Partitioned limits analyzer ready")

        except ImportError:
            print("SKIPPED: partitioned_limits.py not available")
        except Exception as e:
            print(f"ERROR: {e}")

        print("=" * LINE_WIDTH)

    # Significance Testing
    if get_config_bool(config, 'ANALYSIS_OPTIONS',
                       'run_significance_test', False):
        print()
        print("=" * LINE_WIDTH)
        print("SIGNIFICANCE TESTING")
        print("[Hageman & Weis,"
              " https://doi.org/10.1021/acs.analchem.9b01325]")
        print("-" * LINE_WIDTH)

        if not SIGNIFICANCE_AVAILABLE:
            print("SKIPPED: significance.py not available")
        else:
            try:
                sig_alpha = get_config_float(
                    config, 'STATISTICAL_PARAMETERS',
                    'alpha_significance_test', 0.01)

                tester = HybridSignificanceTester(
                    context=context, config=config)

                s_p = tester.calculate_pooled_sd(null_data, uptake_lookup)

                n_typical = context.group_size
                preview_threshold = tester.calculate_row_threshold(
                    s_p, n_typical, n_typical, sig_alpha)

                print(f"Pooled SD: {s_p:.4f} Da")
                print(f"alpha_significance_test = {sig_alpha}")
                print(f"HX significance threshold (Da) ="
                      f" ±{preview_threshold:.3f}")

                results['significance_tester'] = tester
                results['significance_params'] = {
                    's_p': s_p,
                    'alpha': sig_alpha
                }

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

        print("=" * LINE_WIDTH)

    # Cache results
    precomputed_cache[null_exp_path] = results

    return results


# ==============================================================================
# CANDIDATE EVALUATION
# ==============================================================================

def evaluate_protein_section(section_info: Dict, null_exp_results: Dict,
                             config: Dict, loader: DataLoader) -> Dict[str, Any]:
    """
    Evaluate all candidates in a protein section.

    Parameters
    ----------
    section_info : dict
        Section information from get_candidates_from_section()
    null_exp_results : dict
        Results from process_null_experiment()
    config : dict
        Configuration dictionary
    loader : DataLoader
        Data loader instance

    Returns
    -------
    dict
        Evaluation results for all candidates
    """
    results = {'evaluations': {}}

    reference_path = section_info['reference']
    context = null_exp_results['context']
    alpha_ci = context.alpha_ci

    # Get evaluation methods
    eval_methods_str = get_config_value(config, 'ANALYSIS_OPTIONS', 'evaluation_method', 'monte_carlo')
    if eval_methods_str.lower() == 'all':
        eval_methods = list(null_exp_results['limits'].keys())
        if 'partitioned_analyzer' in null_exp_results:
            eval_methods.append('partitioned_limits')
    else:
        eval_methods = [m.strip() for m in eval_methods_str.split(',')]

    # Separate partitioned_limits from regular methods
    regular_methods = [m for m in eval_methods if m != 'partitioned_limits']
    do_partitioned = 'partitioned_limits' in eval_methods and 'partitioned_analyzer' in null_exp_results

    # Load reference
    try:
        reference_df = loader.load_aggregated_data(reference_path)
    except Exception as e:
        print(f"ERROR loading reference: {e}")
        return results

    # Evaluate each candidate
    for cand_num, cand_path in section_info['candidates'].items():
        cand_name = f"candidate_{cand_num}"
        cand_filename = os.path.basename(cand_path).replace('.csv', '')

        print(f"\n  Evaluation of sample: {cand_filename}")
        print()  # Blank line after sample name

        try:
            candidate_df = loader.load_aggregated_data(cand_path)

            results['evaluations'][cand_name] = {}

            # Evaluate with each regular method (global limits)
            for method in regular_methods:
                if method not in null_exp_results['limits']:
                    print(f"    [{method}]: SKIPPED (limits not available)")
                    continue

                limits = null_exp_results['limits'][method]
                eval_result = evaluate_candidate(reference_df, candidate_df, limits, alpha_ci)

                results['evaluations'][cand_name][method] = eval_result

                # Print result with limits
                method_display = method.replace('_', ' ').title()
                print(f"    [{method_display}] limits:")
                print(
                    f"    EAC = {format_report_limit(limits['eac'], config)} Da    "
                    f"|ΔD|_limit = {format_report_limit(limits['delta_d_limit'], config)} Da"
                )
                if eval_result['passed']:
                    print(f"    PASSED by {eval_result['n_pass']}/{eval_result['n_total']} peptide-labeling time combinations")
                    print()  # Blank line after PASSED
                else:
                    print(f"    FAILED by {format_failure_breakdown(eval_result)}")
                    print("    Failed measurements:")
                    for fp in eval_result['failed_peptides']:
                        print_failed_peptide_details(fp)
                    print()  # Blank line after failed peptides

            # Partitioned limits evaluation (special handling)
            if do_partitioned:
                print(f"    [Partitioned] limits:")
                pl_analyzer = null_exp_results['partitioned_analyzer']

                candidate_results = build_candidate_results(
                    reference_df,
                    candidate_df,
                    alpha_ci
                )

                # Build SE lookup for this candidate
                candidate_se_values = {}
                for r in candidate_results:
                    candidate_se_values[(r['Sequence'], r['HX time'])] = r['se']

                # Evaluate by labeling time
                if get_config_bool(config, 'PARTITIONED_LIMITS', 'stratify_by_labeling_time', True):
                    time_limits = pl_analyzer.calculate_limits_by_time()
                    time_eval = pl_analyzer.evaluate_candidate_by_time(candidate_results, time_limits)

                    if time_eval['passed']:
                        print(f"      By labeling time: PASSED ({time_eval['n_pass']}/{time_eval['n_total']})")
                    else:
                        print(f"      By labeling time: FAILED ({time_eval['n_fail']}/{time_eval['n_total']} failed)")

                    results['evaluations'][cand_name]['partitioned_time'] = time_eval

                # Evaluate by peptide length
                if get_config_bool(config, 'PARTITIONED_LIMITS', 'stratify_by_peptide_length', True):
                    length_partitions = get_config_list(config, 'PARTITIONED_LIMITS', 'peptide_length_partition_sizes', [])
                    length_partitions = [int(x) for x in length_partitions] if length_partitions else [6, 7, 8]

                    print(f"      By peptide length:")
                    for partition_width in length_partitions:
                        length_limits, n_empty, empty_list = pl_analyzer.calculate_limits_by_peptide_length(partition_width)

                        if n_empty > 0:
                            total_partitions = len(length_limits) + n_empty
                            print(f"        [WARNING] partition_width={partition_width}: {n_empty}/{total_partitions} partitions are empty")

                        length_eval = pl_analyzer.evaluate_candidate_by_peptide_length(
                            candidate_results, length_limits, partition_width
                        )
                        status = "PASSED" if length_eval['passed'] else f"FAILED ({length_eval['n_fail']})"
                        errors = f" [ERRORS: {len(length_eval['empty_partition_errors'])}]" if length_eval['empty_partition_errors'] else ""
                        print(f"        partition_width={partition_width}: {status}{errors}")

                        results['evaluations'][cand_name][f'partitioned_length_{partition_width}'] = length_eval

                # Evaluate by SE
                if get_config_bool(config, 'PARTITIONED_LIMITS', 'stratify_by_se', True):
                    se_partitions = get_config_list(config, 'PARTITIONED_LIMITS', 'se_n_partitions', [])
                    se_partitions = [int(x) for x in se_partitions] if se_partitions else [70, 80, 90, 100]

                    max_reasonable_partitions = 1000
                    se_partitions_filtered = [n for n in se_partitions if n <= max_reasonable_partitions]

                    print(f"      By standard error:")
                    for n_partitions in se_partitions_filtered:
                        se_limits, n_empty, empty_list = pl_analyzer.calculate_limits_by_se(n_partitions)

                        if n_empty > 0:
                            print(f"        [WARNING] n_partitions={n_partitions}: {n_empty}/{n_partitions} partitions are empty")

                        se_eval = pl_analyzer.evaluate_candidate_by_se(
                            candidate_results, se_limits, n_partitions, candidate_se_values
                        )
                        status = "PASSED" if se_eval['passed'] else f"FAILED ({se_eval['n_fail']})"
                        errors = f" [ERRORS: {len(se_eval['empty_partition_errors'])}]" if se_eval['empty_partition_errors'] else ""
                        print(f"        n_partitions={n_partitions}: {status}{errors}")

                        results['evaluations'][cand_name][f'partitioned_se_{n_partitions}'] = se_eval

            # Significance testing evaluation
            if ('significance_tester' in null_exp_results and
                get_config_bool(config, 'ANALYSIS_OPTIONS',
                                'run_significance_test', False)):

                sig_params = null_exp_results['significance_params']
                tester = null_exp_results['significance_tester']

                sig_results = tester.evaluate_significance(
                    reference_df=reference_df,
                    candidate_df=candidate_df,
                    s_p=sig_params['s_p'],
                    alpha=sig_params['alpha']
                )

                n_sig = sig_results['n_significant']
                n_tot = sig_results['n_total']
                rep_thr = sig_results['representative_threshold']
                expected_n = sig_results['expected_replicates']

                print(f"    [Significance Test]"
                      f" (Hageman et al. https://doi.org/10.1021/acs.analchem.9b01325):")
                print(f"    ΔHX significance threshold (Da) = ±{rep_thr:.3f}"
                      f"    alpha_significance_test = {sig_params['alpha']}")

                total_possible = (
                    n_tot
                    + sig_results['n_skipped_low_n']
                    + sig_results['n_unmatched_ref']
                    + sig_results['n_unmatched_cand']
                )

                if n_sig == 0:
                    print(f"    No significant differences found"
                          f" (0/{n_tot} peptide-time combinations)")
                else:
                    print(f"    {n_sig}/{n_tot} significant"
                          f" differences found")
                    print("    Failed measurements:")
                    for sp in sig_results['significant_peptides']:
                        print_significance_measurement_details(sp)

                print(f"    Evaluated {n_tot}/{total_possible}"
                      f" peptide-time combinations")

                if sig_results['n_skipped_low_n'] > 0:
                    print(f"    WARNING: {sig_results['n_skipped_low_n']}"
                          f" rows were skipped because one side had fewer than"
                          f" {expected_n} replicates.")

                if (sig_results['n_unmatched_ref'] > 0 or
                    sig_results['n_unmatched_cand'] > 0):
                    print(f"    WARNING: {sig_results['n_unmatched_ref']}"
                          f" peptide-time rows were present only in reference"
                          f" and {sig_results['n_unmatched_cand']} only in"
                          f" candidate, so they were excluded.")

                distinct_rep_pairs = {
                    (r['n_ref'], r['n_cand'])
                    for r in sig_results['all_results']
                }
                if len(distinct_rep_pairs) > 1:
                    print("    WARNING: Evaluated rows used mixed replicate-count"
                          " pairs, so the volcano plot lines show only the"
                          " representative threshold for the most common pair.")

                gen_volcano = get_config_bool(
                    config, 'OUTPUT_SETTINGS',
                    'generate_volcano_plots', True)
                save_plots_global = get_config_bool(
                    config, 'OUTPUT_SETTINGS', 'save_plots', True)

                if gen_volcano:
                    fig_format = get_config_value(
                        config, 'OUTPUT_SETTINGS',
                        'figure_format', 'tiff')
                    fig_path = tester.generate_volcano_plot(
                        results=sig_results,
                        output_format=fig_format,
                        save_fig=save_plots_global,
                        show_plot=False,
                        display_name=section_info['display_name'],
                        candidate_name=cand_filename
                    )
                    if fig_path:
                        print(f"    Volcano plot saved: {fig_path}")

                save_sig_csv = get_config_bool(
                    config, 'OUTPUT_SETTINGS',
                    'save_significance_results', True)

                if save_sig_csv:
                    csv_path = tester.save_results_csv(
                        results=sig_results,
                        display_name=section_info['display_name'],
                        candidate_name=cand_filename
                    )
                    if csv_path:
                        print(f"    Results CSV saved: {csv_path}")

                results['evaluations'][cand_name]['significance'] = sig_results
                print()

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    return results


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_pipeline(config_path: str = 'config.txt') -> Dict[str, Any]:
    """
    Run the complete HDX-MS TOST analysis pipeline.

    Parameters
    ----------
    config_path : str
        Path to configuration file

    Returns
    -------
    dict
        All evaluation results
    """

    # ==========================================================================
    # Header
    # ==========================================================================
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print("=" * LINE_WIDTH)
    print("Statistical Evaluation of HDX-MS Data")
    print(f"started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * LINE_WIDTH)

    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    # Parse and validate configuration
    config = parse_config(config_path)

    # Create timestamped output folder
    base_output_dir = get_config_value(config, 'OUTPUT_SETTINGS', 'output_directory', './outputs')
    timestamped_output_dir = os.path.join(base_output_dir, f"run_{run_timestamp}")
    os.makedirs(timestamped_output_dir, exist_ok=True)

    # Update config with timestamped output directory
    if 'OUTPUT_SETTINGS' not in config:
        config['OUTPUT_SETTINGS'] = {}
    config['OUTPUT_SETTINGS']['output_directory'] = timestamped_output_dir

    print(f"Output directory: {timestamped_output_dir}")

    try:
        validate_config(config)
    except ConfigError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Create ExperimentContext from configuration
    context = ExperimentContext.from_config(config)

    # Initialize loader
    loader = DataLoader(config, verbose=context.verbose)

    # Get all protein sections
    protein_sections = get_protein_sections(config)

    if not protein_sections:
        print("ERROR: No PROTEIN_TO_EVALUATE sections found in config")
        sys.exit(1)

    # I. Config validation
    print()  # Blank line after header
    print(f"I. Validation of config file ({config_path}): Success")
    print(f"   Found [{len(protein_sections)}] Protein(s) to evaluate")

    # ==========================================================================
    # Identify unique null experiments (normalize paths to avoid duplicates)
    # ==========================================================================
    null_exp_to_sections = {}
    for section_name in protein_sections:
        section_info = get_candidates_from_section(config, section_name)
        null_exp = section_info['null_experiment']
        # Normalize path to canonical form (handles slashes, case, invisible chars)
        null_exp_key = os.path.normpath(os.path.abspath(null_exp))
        if null_exp_key not in null_exp_to_sections:
            null_exp_to_sections[null_exp_key] = []
        null_exp_to_sections[null_exp_key].append((section_name, section_info))


    # ==========================================================================
    # Process each unique null experiment and its associated protein sections
    # ==========================================================================
    precomputed_cache = {}
    all_results = {}

    for null_exp_path, sections in null_exp_to_sections.items():
        # Create a fresh context for each null experiment (data structure may differ)
        null_context = ExperimentContext.from_config(config)

        # Process null experiment
        null_exp_results = process_null_experiment(
            null_exp_path, null_context, config, loader, precomputed_cache
        )

        # Evaluate each protein section using this null experiment (if enabled)
        run_evaluation = get_config_bool(config, 'ANALYSIS_OPTIONS', 'run_evaluation', True)

        if not run_evaluation:
            # Store results without evaluation
            for section_name, section_info in sections:
                all_results[section_name] = {
                    'display_name': section_info['display_name'],
                    'null_experiment': null_exp_path,
                    'limits': null_exp_results['limits'],
                    'evaluations': {}
                }
            continue

        for section_name, section_info in sections:
            display_name = section_info['display_name']

            print()  # Blank line before ASSESSMENT section
            print("=" * LINE_WIDTH)
            print(f"ASSESSMENT: {display_name}")
            print("=" * LINE_WIDTH)

            if not section_info['candidates']:
                print("  No candidates to evaluate")
                continue

            eval_results = evaluate_protein_section(
                section_info, null_exp_results, config, loader
            )

            all_results[section_name] = {
                'display_name': display_name,
                'null_experiment': null_exp_path,
                **eval_results
            }

    # ==========================================================================
    # Completion
    # ==========================================================================
    print()  # Blank line before completion
    print("=" * LINE_WIDTH)
    print(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * LINE_WIDTH)

    return all_results, timestamped_output_dir


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def main():
    """Main entry point for command line execution."""
    parser = argparse.ArgumentParser(
        description='HDX-MS TOST Equivalence Testing Pipeline'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.txt',
        help='Path to configuration file (default: config.txt)'
    )

    parser.add_argument(
        '--no-pdf',
        action='store_true',
        help='Skip PDF report generation'
    )

    args = parser.parse_args()

    try:
        # Capture all output for PDF report
        with OutputCapture() as capture:
            results, output_dir = run_pipeline(config_path=args.config)

        # Generate PDF report (unless disabled)
        if not args.no_pdf:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_path = os.path.join(output_dir, f"analysis_report_{timestamp}.pdf")
            generated_path = generate_pdf_report(capture.get_output(), pdf_path)
            print(f"Report saved: {generated_path}")

        return 0
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
