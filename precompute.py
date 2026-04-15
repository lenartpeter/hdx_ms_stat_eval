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
HDX-MS EQUIVALENCE TESTING - PRECOMPUTATION MODULE
================================================================================

PURPOSE:
    Handles the precomputation of all statistical values for the HDX-MS TOST
    analysis. For a typical dataset with 8 replicates, 6 labeling times, and
    ~187 peptides, this generates 314,160 pre-computed statistics:

        6 labeling times x 187 peptides x 280 pairings = 314,160 values

    Each precomputed record contains:
    - delta_d: Mean difference in deuterium uptake (sample - reference)
    - ci_lp: Lower bound of the confidence interval
    - ci_up: Upper bound of the confidence interval
    - ci_lp_up_abs_max: Maximum absolute value of CI bounds (for EAC)
    - se: Standard error of the difference
    - sd_ref: Standard deviation of reference group
    - sd_sample: Standard deviation of sample group
    - Detected_Rep_IDs: Comma-separated replicate IDs detected as outliers
    - Removed_Rep_IDs: Comma-separated replicate IDs that were removed
    - N_Valid_Replicates: Number of valid replicates used

INPUTS:
    - ExperimentContext object (from experiment module)
    - Null experiment data (DataFrame and uptake lookup)
    - Replicate pairings

OUTPUTS:
    - Precomputed statistics DataFrame (314k+ records)
    - Settings file for tracking computation parameters
    - Optional outlier report

DEPENDENCIES:
    - config_parser (for configuration access)
    - statistics (for statistical calculations)
    - data_loader (for calculate_peptide_length)
    - experiment (for ExperimentContext)
    - outlier_detection (optional, for outlier filtering)

USAGE:
    from experiment import ExperimentContext
    from precompute import Precomputer

    context = ExperimentContext.from_config(config)
    precomputer = Precomputer(context, config)
    precomputed_df = precomputer.get_or_create_precomputed_data(
        raw_data=null_experiment_df,
        uptake_lookup=lookup_table,
        pairings=all_pairings,
        data_structure=structure,
        null_experiment_file=null_exp_path
    )

================================================================================
"""

import os
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set, TYPE_CHECKING
from tqdm import tqdm
from itertools import combinations

#
from config_parser import (
    get_config_value,
    get_config_bool,
    get_config_float,
    get_config_int
)

from statistics_util import (
    calculate_satterthwaite_df,
    calculate_standard_error,
    calculate_welch_ci,
    get_cached_t_value
)

from data_loader import calculate_peptide_length

# Type hint for ExperimentContext without circular import
if TYPE_CHECKING:
    from experiment import ExperimentContext

# Import outlier detection (optional module)
try:
    from outlier_detection import detect_outliers, AVAILABLE_METHODS, DEFAULT_CUTOFFS
except ImportError:
    # Fallback - outlier detection not available
    AVAILABLE_METHODS = []
    DEFAULT_CUTOFFS = {}
    def detect_outliers(*args, **kwargs):
        raise ImportError("outlier_detection module not available")


# ==============================================================================
# PRECOMPUTER CLASS
# ==============================================================================

class Precomputer:
    """
    Handles precomputation of the 314k statistics dataset.

    This class manages:
    - Computing all delta_D, CI, and SE values for every combination
    - Saving/loading precomputed data to/from CSV
    - Tracking settings used for precomputation
    - Determining when recomputation is needed
    - Stage 0 and Stage 1 outlier filtering

    Parameters
    ----------
    context : ExperimentContext
        Experiment context containing all calculated parameters
    config : dict
        Configuration dictionary (for settings not in context)
    verbose : bool, optional
        If True, print progress messages (default from context)

    Attributes
    ----------
    context : ExperimentContext
        Stored experiment context
    config : dict
        Configuration dictionary
    verbose : bool
        If True, print progress messages
    precomputed_folder : str
        Path to folder for storing precomputed files
    """

    def __init__(self, context: 'ExperimentContext', config: Dict[str, Any],
                 verbose: bool = None):
        """
        Initialize the Precomputer.

        Parameters
        ----------
        context : ExperimentContext
            Experiment context with all parameters
        config : dict
            Configuration dictionary from config.txt
        verbose : bool, optional
            Print progress messages (default from context.verbose)
        """
        self.context = context
        self.config = config
        self.verbose = verbose if verbose is not None else context.verbose

        # 
        self.alpha_ci = context.alpha_ci
        self.group_size = context.group_size
        self.replicates = context.replicates  # List like [1, 2, 3, 4, 5, 6, 7, 8]

        # Precomputation settings from config
        self.precomputed_folder = get_config_value(
            config, 'PRECOMPUTATION', 'precomputed_folder', './precomputed'
        )
        self.force_recompute = get_config_bool(
            config, 'PRECOMPUTATION', 'force_recompute', False
        )

        # Stage 0 outlier settings
        self.stage0_enabled = get_config_bool(config, 'OUTLIER_DETECTION', 'stage0_enabled', False)
        self.stage0_method = get_config_value(config, 'OUTLIER_DETECTION', 'stage0_method', 'boxplot')
        self.stage0_cutoff = get_config_float(config, 'OUTLIER_DETECTION', 'stage0_cutoff', 2.3)

        # Stage 1 outlier settings
        self.stage1_enabled = get_config_bool(config, 'OUTLIER_DETECTION', 'stage1_enabled', False)
        self.stage1_method = get_config_value(config, 'OUTLIER_DETECTION', 'stage1_method', 'boxplot')
        self.stage1_cutoff = get_config_float(config, 'OUTLIER_DETECTION', 'stage1_cutoff', 1.5)
        self.stage1_remove_mode = get_config_value(config, 'OUTLIER_DETECTION', 'stage1_remove_mode', 'single_only')

        # Outlier verbose setting
        self.outlier_verbose = get_config_bool(config, 'OUTLIER_DETECTION', 'verbose', False)

        # Ensure precomputed folder exists
        if not os.path.exists(self.precomputed_folder):
            os.makedirs(self.precomputed_folder)
            self._print(f"Created precomputed folder: {self.precomputed_folder}")

    def _print(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _get_settings_dict(self, null_experiment_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current settings as a dictionary for comparison/storage.

        Parameters
        ----------
        null_experiment_file : str, optional
            Path to the null experiment file for cache change detection

        Returns
        -------
        dict
            Dictionary of all settings that affect precomputation
        """
        settings = {
            'alpha_ci': self.alpha_ci,
            'group_size': self.group_size,
            'n_replicates': len(self.replicates),

            # Stage 0 settings
            'stage0_enabled': self.stage0_enabled,
            'stage0_method': self.stage0_method if self.stage0_enabled else None,
            'stage0_cutoff': self.stage0_cutoff if self.stage0_enabled else None,

            # Stage 1 settings
            'stage1_enabled': self.stage1_enabled,
            'stage1_method': self.stage1_method if self.stage1_enabled else None,
            'stage1_cutoff': self.stage1_cutoff if self.stage1_enabled else None,
            'stage1_remove_mode': self.stage1_remove_mode if self.stage1_enabled else None,

            'created_at': datetime.now().isoformat()
        }

        if null_experiment_file:
            settings['null_experiment_file'] = os.path.abspath(null_experiment_file)
            settings['null_experiment_hash'] = self._get_file_hash(null_experiment_file)

        return settings

    def _get_file_hash(self, file_path: str) -> str:
        """
        Calculate MD5 hash of a file for change detection.

        Parameters
        ----------
        file_path : str
            Path to the file

        Returns
        -------
        str
            MD5 hash string
        """
        if not os.path.exists(file_path):
            return ""

        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _get_precomputed_paths(self, null_experiment_file: str) -> Tuple[str, str, str]:
        """
        Generate paths for precomputed CSV, settings, and outlier report files.

        Parameters
        ----------
        null_experiment_file : str
            Path to the null experiment file

        Returns
        -------
        tuple
            (csv_path, settings_path, outlier_report_path)
        """
        # Use base filename without extension
        base_name = os.path.splitext(os.path.basename(null_experiment_file))[0]

        csv_path = os.path.join(self.precomputed_folder, f"{base_name}_precomputed.csv")
        settings_path = os.path.join(self.precomputed_folder, f"{base_name}_precomputed_settings.txt")
        outlier_report_path = os.path.join(self.precomputed_folder, f"{base_name}_outlier_report.csv")

        return csv_path, settings_path, outlier_report_path

    def _settings_match(self, settings_path: str,
                        null_experiment_file: Optional[str] = None) -> bool:
        """
        Check if saved settings match current settings.

        Parameters
        ----------
        settings_path : str
            Path to the settings file
        null_experiment_file : str, optional
            Path to the null experiment file for cache change detection

        Returns
        -------
        bool
            True if settings match, False otherwise
        """
        if not os.path.exists(settings_path):
            return False

        try:
            with open(settings_path, 'r') as f:
                saved_settings = json.load(f)
        except (json.JSONDecodeError, IOError):
            return False

        current_settings = self._get_settings_dict(null_experiment_file)

        # Compare relevant settings (excluding timestamp)
        keys_to_compare = [
            'alpha_ci', 'group_size', 'n_replicates',
            'stage0_enabled', 'stage0_method', 'stage0_cutoff',
            'stage1_enabled', 'stage1_method', 'stage1_cutoff', 'stage1_remove_mode'
        ]

        for key in keys_to_compare:
            if saved_settings.get(key) != current_settings.get(key):
                self._print(f"    Settings mismatch: {key}")
                self._print(f"      Saved: {saved_settings.get(key)}")
                self._print(f"      Current: {current_settings.get(key)}")
                return False

        if null_experiment_file:
            saved_hash = saved_settings.get('null_experiment_hash')
            current_hash = current_settings.get('null_experiment_hash')

            if not saved_hash:
                self._print("    Settings mismatch: null_experiment_hash")
                self._print("      Saved: missing (legacy cache metadata)")
                self._print(f"      Current: {current_hash}")
                return False

            if saved_hash != current_hash:
                self._print("    Settings mismatch: null_experiment_hash")
                self._print(f"      Saved: {saved_hash}")
                self._print(f"      Current: {current_hash}")
                return False

        return True

    def _save_settings(self, settings_path: str,
                       null_experiment_file: Optional[str] = None) -> None:
        """
        Save current settings to file.

        Parameters
        ----------
        settings_path : str
            Path to save settings
        null_experiment_file : str, optional
            Path to the null experiment file for cache change detection
        """
        settings = self._get_settings_dict(null_experiment_file)

        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)

        self._print(f"    Saved settings to: {settings_path}")

    # ==========================================================================
    # STAGE 0: UPTAKE RANGE FILTERING
    # ==========================================================================

    def _calculate_uptake_ranges(self, uptake_lookup: Dict[Tuple, float],
                                  peptides: List[str],
                                  time_points: List[float]) -> Dict[Tuple[str, float], Dict]:
        """
        Calculate uptake range statistics for each peptide-time combination.

        Parameters
        ----------
        uptake_lookup : dict
            Mapping of (Sequence, Replicate, Time) -> Uptake
        peptides : list
            List of peptide sequences
        time_points : list
            List of labeling times

        Returns
        -------
        dict
            Mapping of (Sequence, Time) -> {uptakes, range, n_reps, mean, std}
        """
        range_data = {}

        for peptide in peptides:
            for time_point in time_points:
                uptakes = []
                # 
                for rep in self.replicates:
                    key = (peptide, rep, time_point)
                    if key in uptake_lookup:
                        uptakes.append(uptake_lookup[key])

                if len(uptakes) >= 2:
                    range_data[(peptide, time_point)] = {
                        'uptakes': uptakes,
                        'range': max(uptakes) - min(uptakes),
                        'n_reps': len(uptakes),
                        'mean': np.mean(uptakes),
                        'std': np.std(uptakes, ddof=1) if len(uptakes) > 1 else 0
                    }

        return range_data

    def _apply_stage0_filtering(self, uptake_lookup: Dict[Tuple, float],
                                 peptides: List[str],
                                 time_points: List[float]) -> Tuple[Set[Tuple[str, float]], List[Dict]]:
        """
        Apply Stage 0 filtering: identify peptide-time combinations with outlier uptake ranges.

        Parameters
        ----------
        uptake_lookup : dict
            Mapping of (Sequence, Replicate, Time) -> Uptake
        peptides : list
            List of peptide sequences
        time_points : list
            List of labeling times

        Returns
        -------
        tuple
            (excluded_peptide_times, stage0_report)
            - excluded_peptide_times: Set of (peptide, time) tuples to exclude
            - stage0_report: List of report entries for excluded items
        """
        self._print(f"\n  Stage 0 Configuration:")
        self._print(f"    Method: {self.stage0_method}")
        self._print(f"    Cutoff: {self.stage0_cutoff}")

        # Calculate uptake ranges for all peptide-times
        range_data = self._calculate_uptake_ranges(uptake_lookup, peptides, time_points)

        if not range_data:
            self._print("    No valid peptide-time combinations found")
            return set(), []

        # Extract ranges for outlier detection
        peptide_times = list(range_data.keys())
        ranges = np.array([range_data[pt]['range'] for pt in peptide_times])

        self._print(f"    Analyzing {len(ranges)} peptide-time combinations")
        self._print(f"    Uptake range: min={ranges.min():.4f}, max={ranges.max():.4f}, median={np.median(ranges):.4f}")

        # Detect outliers
        result = detect_outliers(
            ranges,
            method=self.stage0_method,
            cutoff=self.stage0_cutoff,
            verbose=self.outlier_verbose
        )

        if result is None:
            self._print("    Outlier detection failed (insufficient data)")
            return set(), []

        # Build exclusion set and report
        excluded_peptide_times = set()
        stage0_report = []

        for idx in result['outlier_indices']:
            peptide, time_point = peptide_times[idx]
            excluded_peptide_times.add((peptide, time_point))

            data = range_data[(peptide, time_point)]
            stage0_report.append({
                'Stage': 'Stage0',
                'Sequence': peptide,
                'HX_time_sec': time_point,
                'Reason': 'Uptake range outlier',
                'Uptake_Range': data['range'],
                'N_Replicates': data['n_reps'],
                'Mean_Uptake': data['mean'],
                'Std_Uptake': data['std'],
                'Detected_Rep_IDs': '',
                'Removed_Rep_IDs': '',
                'N_Valid_Replicates': 0,
                'Method': self.stage0_method,
                'Cutoff': self.stage0_cutoff,
                'Lower_Bound': result.get('lower_bound', np.nan),
                'Upper_Bound': result.get('upper_bound', np.nan)
            })

        self._print(f"    Outliers detected: {result['n_outliers']} ({result['outlier_percentage']:.1f}%)")

        if self.outlier_verbose and result['n_outliers'] > 0:
            self._print(f"    Excluded peptide-times:")
            for entry in stage0_report:
                self._print(
                    f"      {entry['Sequence']} @ {entry['HX_time_sec']}s "
                    f"(range={entry['Uptake_Range']:.4f})"
                )

        return excluded_peptide_times, stage0_report

    # ==========================================================================
    # STAGE 1: REPLICATE FILTERING
    # ==========================================================================

    def _get_valid_replicates_stage1(self, uptake_values: List[float],
                                      replicate_ids: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """
        Apply Stage 1 filtering for a single peptide-time combination.

        Parameters
        ----------
        uptake_values : list
            Uptake values for available replicates
        replicate_ids : list
            Corresponding replicate IDs

        Returns
        -------
        tuple
            (valid_replicates, detected_outliers, removed_replicates)
        """
        if len(uptake_values) < 4:
            # Not enough data for outlier detection
            return replicate_ids.copy(), [], []

        # Detect outliers
        result = detect_outliers(
            uptake_values,
            method=self.stage1_method,
            cutoff=self.stage1_cutoff,
            verbose=False  # Too noisy for per-peptide-time
        )

        if result is None:
            return replicate_ids.copy(), [], []

        # Map outlier indices back to replicate IDs
        detected_outliers = [replicate_ids[idx] for idx in result['outlier_indices']]
        n_outliers = len(detected_outliers)

        # Determine which to remove based on mode
        removed_replicates = []

        if self.stage1_remove_mode == 'none':
            # Detect but don't remove
            pass
        elif self.stage1_remove_mode == 'single_only':
            # Remove only if exactly 1 outlier
            if n_outliers == 1:
                removed_replicates = detected_outliers.copy()
        elif self.stage1_remove_mode == 'up_to_two':
            # Remove if 1 or 2 outliers
            if n_outliers in [1, 2]:
                removed_replicates = detected_outliers.copy()

        # Calculate valid replicates
        valid_replicates = [r for r in replicate_ids if r not in removed_replicates]

        return valid_replicates, detected_outliers, removed_replicates

    def _generate_pairings_for_replicates(self, valid_replicates: List[int]) -> List[Dict]:
        """
        Generate all valid group_size-vs-group_size pairings from available replicates.

        Parameters
        ----------
        valid_replicates : list
            List of valid replicate IDs

        Returns
        -------
        list
            List of pairing dicts with 'pairing_id', 'ref_reps', 'sample_reps'
        """
        n = len(valid_replicates)
        min_required = 2 * self.group_size

        if n < min_required:
            return []

        pairings = []
        pairing_id = 0

        for ref_combo in combinations(valid_replicates, self.group_size):
            remaining = [r for r in valid_replicates if r not in ref_combo]
            for sample_combo in combinations(remaining, self.group_size):
                # Use canonical ordering to avoid duplicates
                ref_sorted = tuple(sorted(ref_combo))
                sample_sorted = tuple(sorted(sample_combo))

                if ref_sorted < sample_sorted:
                    pairings.append({
                        'pairing_id': pairing_id,
                        'ref_reps': list(ref_sorted),
                        'sample_reps': list(sample_sorted)
                    })
                    pairing_id += 1

        return pairings

    # ==========================================================================
    # OUTLIER REPORT GENERATION
    # ==========================================================================

    def _generate_outlier_report(self, stage0_report: List[Dict],
                                  stage1_report: List[Dict],
                                  outlier_report_path: str) -> None:
        """
        Generate combined outlier report CSV.

        Parameters
        ----------
        stage0_report : list
            Report entries from Stage 0
        stage1_report : list
            Report entries from Stage 1
        outlier_report_path : str
            Path to save the report
        """
        all_entries = stage0_report + stage1_report

        if not all_entries:
            self._print("  No outliers to report")
            return

        # Define column order
        columns = [
            'Stage', 'Sequence', 'HX_time_sec', 'Reason',
            'Detected_Rep_IDs', 'Removed_Rep_IDs', 'N_Valid_Replicates',
            'Uptake_Range', 'N_Replicates', 'Mean_Uptake', 'Std_Uptake',
            'Method', 'Cutoff', 'Lower_Bound', 'Upper_Bound'
        ]

        # Create DataFrame
        df = pd.DataFrame(all_entries)

        # Reorder columns (only include those that exist)
        existing_cols = [c for c in columns if c in df.columns]
        df = df[existing_cols]

        # Sort by Stage, then Sequence, then Time
        df = df.sort_values(['Stage', 'Sequence', 'HX_time_sec'])

        # Save
        df.to_csv(outlier_report_path, index=False)
        self._print(f"  Outlier report saved: {outlier_report_path}")
        self._print(f"    Stage 0 entries: {len(stage0_report)}")
        self._print(f"    Stage 1 entries: {len(stage1_report)}")

    # ==========================================================================
    # MAIN COMPUTATION
    # ==========================================================================

    def compute_all_statistics(self, raw_data: pd.DataFrame,
                               uptake_lookup: Dict[Tuple, float],
                               pairings: List[Dict],
                               time_points: List[float],
                               peptides: List[str],
                               null_experiment_file: str = None) -> pd.DataFrame:
        """
        Compute delta_D, CI, and SE for all peptide-time-pairing combinations.

        This is the main precomputation function that generates the 314k dataset,
        with optional Stage 0 and Stage 1 outlier filtering.

        Parameters
        ----------
        raw_data : pd.DataFrame
            The null experiment data (for reference, not directly used)
        uptake_lookup : dict
            Dictionary mapping (Sequence, Replicate, Time) -> Uptake
        pairings : list
            List of pairing dictionaries from generate_all_pairings()
            (used only when Stage 1 is disabled)
        time_points : list
            List of labeling times
        peptides : list
            List of peptide sequences
        null_experiment_file : str, optional
            Path to null experiment file (for outlier report naming)

        Returns
        -------
        pd.DataFrame
            Precomputed statistics with columns:
            Time_sec, Sequence, Peptide_Length, Pairing_ID,
            delta_d, ci_lp, ci_up, ci_lp_up_abs_max, se, sd_ref, sd_sample,
            Detected_Rep_IDs, Removed_Rep_IDs, N_Valid_Replicates
        """
        self._print("Computing all statistics...")

        stage0_report = []
        stage1_report = []
        excluded_peptide_times = set()

        # ======================================================================
        # STAGE 0: Filter peptide-time combinations by uptake range
        # ======================================================================
        if self.stage0_enabled:
            self._print("\n" + "-"*70)
            self._print("STAGE 0: UPTAKE RANGE FILTERING")
            self._print("-"*70)

            excluded_peptide_times, stage0_report = self._apply_stage0_filtering(
                uptake_lookup, peptides, time_points
            )

            self._print(f"  Excluded: {len(excluded_peptide_times)} peptide-time combinations")

        # ======================================================================
        # MAIN COMPUTATION LOOP (with optional Stage 1)
        # ======================================================================

        # Calculate expected combinations
        n_peptide_times = len(peptides) * len(time_points) - len(excluded_peptide_times)

        if self.stage1_enabled:
            self._print(f"\n  Stage 1 Configuration:")
            self._print(f"    Method: {self.stage1_method}")
            self._print(f"    Cutoff: {self.stage1_cutoff}")
            self._print(f"    Remove mode: {self.stage1_remove_mode}")
            self._print(f"\n  Processing {n_peptide_times} peptide-time combinations with Stage 1 filtering...")
        else:
            total_calculations = n_peptide_times * len(pairings)
            self._print(f"\n  Total combinations: {len(time_points)} times x "
                       f"{len(peptides)} peptides x {len(pairings)} pairings = {total_calculations:,}")

        results_list = []
        skipped_count = 0
        stage1_outlier_count = 0
        stage1_skip_count = 0

        # Progress bar
        with tqdm(total=n_peptide_times, desc="  Computing", unit="pep-time",
                  disable=not self.verbose) as pbar:

            for time_point in time_points:
                for peptide in peptides:
                    # Check Stage 0 exclusion
                    if (peptide, time_point) in excluded_peptide_times:
                        continue

                    peptide_length = calculate_peptide_length(peptide)

                    # Get available replicates for this peptide-time
                    # 
                    available_reps = []
                    uptake_values = []
                    for rep in self.replicates:
                        key = (peptide, rep, time_point)
                        if key in uptake_lookup:
                            available_reps.append(rep)
                            uptake_values.append(uptake_lookup[key])

                    # ===========================================================
                    # STAGE 1: Detect/remove outlier replicates
                    # ===========================================================
                    detected_outliers = []
                    removed_replicates = []
                    valid_replicates = available_reps

                    if self.stage1_enabled and len(uptake_values) >= 4:
                        valid_replicates, detected_outliers, removed_replicates = \
                            self._get_valid_replicates_stage1(uptake_values, available_reps)

                        # Track for report and statistics
                        if detected_outliers:
                            stage1_outlier_count += 1
                            stage1_report.append({
                                'Stage': 'Stage1',
                                'Sequence': peptide,
                                'HX_time_sec': time_point,
                                'Detected_Rep_IDs': ','.join(map(str, sorted(detected_outliers))),
                                'Removed_Rep_IDs': ','.join(map(str, sorted(removed_replicates))),
                                'N_Valid_Replicates': len(valid_replicates),
                                'Reason': 'Outlier replicate(s)' if removed_replicates else 'Detected but not removed',
                                'Method': self.stage1_method,
                                'Cutoff': self.stage1_cutoff
                            })
                            if self.outlier_verbose:
                                detected_str = ','.join(map(str, sorted(detected_outliers)))
                                removed_str = ','.join(map(str, sorted(removed_replicates))) if removed_replicates else 'none'
                                self._print(
                                    f"    Stage 1 outlier: {peptide} @ {time_point}s "
                                    f"(detected={detected_str}; removed={removed_str}; "
                                    f"valid_replicates={len(valid_replicates)})"
                                )

                    # Check if enough replicates for group_size-vs-group_size pairings
                    min_required = 2 * self.group_size
                    if len(valid_replicates) < min_required:
                        stage1_skip_count += 1
                        stage1_report.append({
                            'Stage': 'Stage1_Skip',
                            'Sequence': peptide,
                            'HX_time_sec': time_point,
                            'Detected_Rep_IDs': ','.join(map(str, sorted(detected_outliers))),
                            'Removed_Rep_IDs': ','.join(map(str, sorted(removed_replicates))),
                            'N_Valid_Replicates': len(valid_replicates),
                            'Reason': f'Insufficient replicates ({len(valid_replicates)}<{min_required})',
                            'Method': self.stage1_method if self.stage1_enabled else '',
                            'Cutoff': self.stage1_cutoff if self.stage1_enabled else ''
                        })
                        pbar.update(1)
                        continue

                    # Get applicable pairings
                    if self.stage1_enabled and len(valid_replicates) < len(self.replicates):
                        # Generate pairings for reduced replicate set
                        applicable_pairings = self._generate_pairings_for_replicates(valid_replicates)
                    else:
                        # Use full pairing set
                        applicable_pairings = pairings

                    # Prepare outlier tracking strings
                    detected_str = ','.join(map(str, sorted(detected_outliers))) if detected_outliers else ''
                    removed_str = ','.join(map(str, sorted(removed_replicates))) if removed_replicates else ''

                    # ===========================================================
                    # Compute statistics for each pairing
                    # ===========================================================
                    for pairing in applicable_pairings:
                        pairing_id = pairing['pairing_id']
                        ref_reps = pairing['ref_reps']
                        sample_reps = pairing['sample_reps']

                        # Get uptake values for reference group
                        uptakes_ref = []
                        for rep in ref_reps:
                            key = (peptide, rep, time_point)
                            if key in uptake_lookup:
                                uptakes_ref.append(uptake_lookup[key])

                        # Get uptake values for sample group
                        uptakes_sample = []
                        for rep in sample_reps:
                            key = (peptide, rep, time_point)
                            if key in uptake_lookup:
                                uptakes_sample.append(uptake_lookup[key])

                        # Skip if insufficient data
                        if len(uptakes_ref) < 2 or len(uptakes_sample) < 2:
                            skipped_count += 1
                            continue

                        # Calculate statistics using imported function
                        result = calculate_welch_ci(
                            np.array(uptakes_ref),
                            np.array(uptakes_sample),
                            alpha_ci=self.alpha_ci
                        )

                        # Store result with outlier tracking
                        results_list.append({
                            'Time_sec': time_point,
                            'Sequence': peptide,
                            'Peptide_Length': peptide_length,
                            'Pairing_ID': pairing_id,
                            'delta_d': result['delta_d'],
                            'ci_lp': result['ci_lp'],
                            'ci_up': result['ci_up'],
                            'ci_lp_up_abs_max': result['ci_lp_up_abs_max'],
                            'se': result['se'],
                            'sd_ref': result['sd_ref'],
                            'sd_sample': result['sd_sample'],
                            'Detected_Rep_IDs': detected_str,
                            'Removed_Rep_IDs': removed_str,
                            'N_Valid_Replicates': len(valid_replicates)
                        })

                    pbar.update(1)

        # Create DataFrame
        precomputed_df = pd.DataFrame(results_list)

        # Print summary
        self._print(f"\n  Computed: {len(precomputed_df):,} statistics")
        if skipped_count > 0:
            self._print(f"  Skipped: {skipped_count:,} (insufficient data)")

        if self.stage0_enabled:
            self._print(f"  Stage 0: Excluded {len(stage0_report)} peptide-time combinations")

        if self.stage1_enabled:
            self._print(f"  Stage 1: Found outliers in {stage1_outlier_count} peptide-time combinations")
            self._print(f"  Stage 1: Skipped {stage1_skip_count} peptide-time combinations (<{2*self.group_size} valid replicates)")

        # Generate outlier report
        if (self.stage0_enabled or self.stage1_enabled) and null_experiment_file:
            _, _, outlier_report_path = self._get_precomputed_paths(null_experiment_file)
            self._generate_outlier_report(stage0_report, stage1_report, outlier_report_path)

        self._print("="*70)

        return precomputed_df

    def save_precomputed(self, df: pd.DataFrame, csv_path: str, settings_path: str,
                         null_experiment_file: Optional[str] = None) -> None:
        """
        Save precomputed DataFrame and settings to files.

        Parameters
        ----------
        df : pd.DataFrame
            Precomputed statistics
        csv_path : str
            Path for CSV file
        settings_path : str
            Path for settings file
        null_experiment_file : str, optional
            Path to the null experiment file for cache change detection
        """
        self._print(f"\n  Saving precomputed data...")

        # Save CSV
        df.to_csv(csv_path, index=False)
        file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        self._print(f"    CSV saved: {csv_path}")
        self._print(f"    File size: {file_size_mb:.1f} MB")

        # Save settings
        self._save_settings(settings_path, null_experiment_file)

    def load_precomputed(self, csv_path: str) -> pd.DataFrame:
        """
        Load precomputed DataFrame from CSV.

        Parameters
        ----------
        csv_path : str
            Path to CSV file

        Returns
        -------
        pd.DataFrame
            Loaded precomputed statistics
        """
        df = pd.read_csv(csv_path)
        return df

    def get_or_create_precomputed_data(self, raw_data: pd.DataFrame,
                                       uptake_lookup: Dict[Tuple, float],
                                       pairings: List[Dict],
                                       data_structure: Dict[str, Any],
                                       null_experiment_file: str) -> pd.DataFrame:
        """
        Get precomputed data, creating it if necessary.

        This is the main entry point for obtaining precomputed statistics.
        It checks if valid precomputed data exists and matches current settings,
        and only recomputes if necessary.

        Parameters
        ----------
        raw_data : pd.DataFrame
            The null experiment data
        uptake_lookup : dict
            Dictionary mapping (Sequence, Replicate, Time) -> Uptake
        pairings : list
            List of pairing dictionaries
        data_structure : dict
            Dictionary with time_points, peptides, etc.
        null_experiment_file : str
            Path to the null experiment file (for naming)

        Returns
        -------
        pd.DataFrame
            Precomputed statistics DataFrame
        """
        csv_path, settings_path, _ = self._get_precomputed_paths(null_experiment_file)

        # Check if we need to recompute
        need_recompute = False

        if self.force_recompute:
            need_recompute = True
        elif not os.path.exists(csv_path):
            need_recompute = True
        elif not os.path.exists(settings_path):
            need_recompute = True
        elif not self._settings_match(settings_path, null_experiment_file):
            need_recompute = True

        if need_recompute:
            # Compute all statistics
            precomputed_df = self.compute_all_statistics(
                raw_data=raw_data,
                uptake_lookup=uptake_lookup,
                pairings=pairings,
                time_points=data_structure['time_points'],
                peptides=data_structure['peptides'],
                null_experiment_file=null_experiment_file
            )

            # Save to files
            self.save_precomputed(
                precomputed_df,
                csv_path,
                settings_path,
                null_experiment_file
            )
        else:
            # Load existing data - print status message
            print("Precomputed data status: already generated")
            print(f"  CSV found at: {csv_path}")
            print(f"  Settings file found at:")
            print(f"{settings_path}")
            precomputed_df = pd.read_csv(csv_path)
            print(f"Loaded: {len(precomputed_df):,} records")

        return precomputed_df

    def get_global_limits(self, precomputed_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate global limits from precomputed data.

        Parameters
        ----------
        precomputed_df : pd.DataFrame
            Precomputed statistics

        Returns
        -------
        dict
            Dictionary with:
            - delta_d_limit: Maximum |delta_D| observed
            - eac: Maximum |CI_LP,UP| observed (Equivalence Acceptance Criterion)
            - n_values: Total number of values examined
        """
        delta_d_abs_max = precomputed_df['delta_d'].abs().max()
        ci_lp_up_abs_max = precomputed_df['ci_lp_up_abs_max'].max()

        limits = {
            'delta_d_limit': float(delta_d_abs_max),
            'eac': float(ci_lp_up_abs_max),
            'n_values': len(precomputed_df)
        }

        return limits

    def get_uptake_ranges(self, raw_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate uptake ranges for each peptide-time combination.

        The uptake range is the difference between max and min uptake values
        across all replicates for each peptide-labeling time combination.

        Parameters
        ----------
        raw_df : pd.DataFrame
            Raw data DataFrame with columns: Sequence, HX time, Replicate number, Uptake (Da)

        Returns
        -------
        np.ndarray
            Array of uptake range values (one per peptide-time combination)
        """
        # Build uptake lookup: (Sequence, Replicate, Time) -> Uptake
        uptake_lookup = {}
        for _, row in raw_df.iterrows():
            key = (row['Sequence'], int(row['Replicate number']), float(row['HX time']))
            uptake_lookup[key] = float(row['Uptake (Da)'])

        # Get unique peptides and time points
        peptides = sorted(raw_df['Sequence'].unique())
        time_points = sorted(raw_df['HX time'].unique())

        # Calculate ranges
        range_data = self._calculate_uptake_ranges(uptake_lookup, peptides, time_points)

        # Extract just the range values as an array
        ranges = np.array([data['range'] for data in range_data.values()])

        return ranges

    def generate_uptake_range_figure(self, raw_df: pd.DataFrame,
                                      output_format: str = None,
                                      show_plot: bool = False) -> str:
        """
        Generate and save the uptake range scatter plot.

        Parameters
        ----------
        raw_df : pd.DataFrame
            Raw data DataFrame
        output_format : str, optional
            Figure format: 'tiff' (default), 'png', or 'pdf'
        show_plot : bool
            Whether to display the plot

        Returns
        -------
        str
            Path to saved figure, or None if not saved
        """
        from plotting import plot_uptake_ranges, get_format_extension, DEFAULT_FORMAT
        from datetime import datetime

        if output_format is None:
            output_format = DEFAULT_FORMAT

        # Get uptake ranges
        ranges = self.get_uptake_ranges(raw_df)

        if len(ranges) == 0:
            self._print("No uptake ranges to plot")
            return None

        # Get output directory from config
        output_dir = get_config_value(self.config, 'OUTPUT_SETTINGS', 'output_directory', './outputs')

        # Generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = get_format_extension(output_format)
        output_path = os.path.join(output_dir, f"uptake_range_dotplot_{timestamp}{ext}")

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate plot
        self._print(f"\nGenerating uptake range figure...")
        self._print(f"  Peptide-time combinations: {len(ranges)}")
        self._print(f"  Range: min={ranges.min():.4f}, max={ranges.max():.4f}, median={np.median(ranges):.4f}")

        plot_uptake_ranges(
            uptake_ranges=ranges,
            output_path=output_path,
            output_format=output_format,
            show_plot=show_plot,
            verbose=False,  # main.py handles the output message
            config=self.config
        )

        return output_path


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def build_max_arrays(precomputed_df: pd.DataFrame,
                     time_points: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build arrays of max |delta_D| and max |CI_LP,UP| per time-pairing combination.

    This function collapses the precomputed data by taking the maximum across
    peptides for each (time, pairing) combination. Used for Monte Carlo and
    Direct Percentile methods.

    Parameters
    ----------
    precomputed_df : pd.DataFrame
        Precomputed statistics DataFrame
    time_points : list
        Sorted list of labeling times

    Returns
    -------
    tuple
        (max_delta_d, max_ci_lp_up) arrays of shape (n_times, n_pairings)
    """
    # Group by time and pairing, take max across peptides
    grouped = precomputed_df.groupby(
        ['Time_sec', 'Pairing_ID'],
        as_index=False
    ).agg({
        'delta_d': lambda x: np.abs(x).max(),  # max |delta_D|
        'ci_lp_up_abs_max': 'max'              # max |CI_LP,UP|
    })

    # Build arrays
    time_to_idx = {t: i for i, t in enumerate(time_points)}
    n_times = len(time_points)

    # Get max pairing ID to determine array size
    max_pairing_id = int(grouped['Pairing_ID'].max())
    n_pairings = max_pairing_id + 1

    max_delta_d = np.full((n_times, n_pairings), np.nan, dtype=float)
    max_ci_lp_up = np.full((n_times, n_pairings), np.nan, dtype=float)

    for _, row in grouped.iterrows():
        t_idx = time_to_idx[row['Time_sec']]
        p_idx = int(row['Pairing_ID'])
        max_delta_d[t_idx, p_idx] = float(row['delta_d'])
        max_ci_lp_up[t_idx, p_idx] = float(row['ci_lp_up_abs_max'])

    return max_delta_d, max_ci_lp_up


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    print("PRECOMPUTE MODULE - TEST")
    print("Run main.py to execute the full pipeline.")
