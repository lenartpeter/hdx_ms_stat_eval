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
HDX-MS EQUIVALENCE TESTING - DATA LOADER MODULE
================================================================================

PURPOSE:
    Handles all data loading and validation operations for the HDX-MS
    analysis pipeline. This module provides functionality to:

    1. Load raw HDX-MS data from CSV files (two formats supported)
    2. Validate data structure against expected configuration
    3. Create efficient lookup tables for O(1) access to uptake values
    4. Generate all non-overlapping replicate pairings (parameterized)
    5. Load and validate biosimilar/mock candidate data pairs

INPUTS:
    - CSV files containing HDX-MS data
    - Configuration dictionary (from config_parser module)
    - ExperimentContext (from experiment module) for validation parameters

OUTPUTS:
    - Validated pandas DataFrames
    - Uptake lookup dictionaries
    - Replicate pairing lists
    - Data structure information

DEPENDENCIES:
    - config_parser (for configuration access - single source of truth)
    - pandas, numpy (data handling)

DATA FORMATS:
-------------
1. NULL EXPERIMENT FORMAT (individual replicates):
   Columns: Sequence, Replicate number, HX time, Uptake (Da)
   Each row = one replicate measurement

2. AGGREGATED FORMAT (for biosimilar/reference comparison):
   Columns: Protein state, Sequence, HX time, Uptake (Da), Uptake SD (Da), Replicate number
   Each row = averaged data across replicates
   "Replicate number" here means the COUNT of replicates used for averaging

USAGE:
    from config_parser import parse_config, validate_config, get_config_int
    from data_loader import DataLoader, validate_null_experiment_data

    config = parse_config("config.txt")
    validate_config(config)

    loader = DataLoader(config, verbose=True)
    df, structure = loader.load_null_experiment(null_exp_path)

    # Validate data matches expected configuration
    n_replicates = get_config_int(config, 'STATISTICAL_PARAMETERS', 'n_replicates', 8)
    validate_null_experiment_data(df, n_replicates)

    # Generate pairings using configuration parameters
    group_size = get_config_int(config, 'STATISTICAL_PARAMETERS', 'group_size', 3)
    pairings = loader.generate_all_pairings(structure['replicates'], group_size)

================================================================================
"""

import os
import re
import pandas as pd
import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Any

# Import configuration helpers from single source of truth
from config_parser import (
    get_config_value,
    get_config_bool,
    get_config_float,
    get_config_int,
    get_config_list,
    get_protein_sections,
    get_candidates_from_section
)


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Required columns for null experiment data (individual replicates)
NULL_EXPERIMENT_COLUMNS = ['Sequence', 'Replicate number', 'HX time', 'Uptake (Da)']

# Required columns for aggregated data (biosimilar/reference)
AGGREGATED_COLUMNS = ['Sequence', 'HX time', 'Uptake (Da)', 'Uptake SD (Da)', 'Replicate number']

# Column name mappings for standardization (empty - no aliases currently defined)
COLUMN_ALIASES = {}

# ==============================================================================
# DATA VALIDATION FUNCTIONS
# ==============================================================================

def validate_null_experiment_data(df: pd.DataFrame, expected_replicates: int) -> None:
    """
    Validate null experiment data structure against expected configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Null experiment dataframe to validate
    expected_replicates : int
        Expected number of replicates from config (n_replicates)

    Raises
    ------
    ValueError
        If validation fails, with descriptive error message listing all issues

    Notes
    -----
    Checks performed:
    - Required columns exist
    - Replicate numbers match expected range [1, n_replicates]
    - No missing values in critical columns
    - Data types are correct
    """
    errors = []

    # Check required columns
    missing = [col for col in NULL_EXPERIMENT_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    # Check replicate numbers
    if 'Replicate number' in df.columns:
        actual_reps = sorted(df['Replicate number'].unique().tolist())
        expected_reps = list(range(1, expected_replicates + 1))

        if actual_reps != expected_reps:
            errors.append(
                f"Replicate number mismatch:\n"
                f"    Expected: {expected_reps}\n"
                f"    Found: {actual_reps}"
            )

    # Check for missing values in critical columns
    for col in NULL_EXPERIMENT_COLUMNS:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                errors.append(f"Column '{col}' has {null_count} missing values")

    # Check data types
    if 'Uptake (Da)' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['Uptake (Da)']):
            errors.append("Column 'Uptake (Da)' must be numeric")

    if 'HX time' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['HX time']):
            errors.append("Column 'HX time' must be numeric")

    if errors:
        raise ValueError(
            "Null experiment data validation failed:\n  " + "\n  ".join(errors)
        )


def validate_aggregated_data(df: pd.DataFrame) -> None:
    """
    Validate aggregated (biosimilar/reference) data structure.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated dataframe to validate

    Raises
    ------
    ValueError
        If validation fails, with descriptive error message listing all issues

    Notes
    -----
    Checks performed:
    - Required columns exist
    - No missing values in critical columns
    - Numeric columns have correct data types
    - Replicate counts are positive
    """
    errors = []

    # Check required columns
    missing = [col for col in AGGREGATED_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    # Check for missing values in critical columns
    for col in AGGREGATED_COLUMNS:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                errors.append(f"Column '{col}' has {null_count} missing values")

    # Check data types
    numeric_cols = ['Uptake (Da)', 'Uptake SD (Da)', 'HX time', 'Replicate number']
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' must be numeric")

    # Check replicate counts are positive
    if 'Replicate number' in df.columns:
        if (df['Replicate number'] <= 0).any():
            errors.append("'Replicate number' (count) must be positive")

    # Check replicate counts are integer-valued
    if 'Replicate number' in df.columns:
        rep_counts = df['Replicate number']
        non_integer_mask = ~np.isclose(rep_counts, np.round(rep_counts))
        if non_integer_mask.any():
            bad_values = rep_counts[non_integer_mask].unique()[:3]
            errors.append(
                "'Replicate number' (count) must contain integer values "
                f"(e.g. found {bad_values.tolist()})"
            )

    # Check (Sequence, HX time) uniqueness
    if 'Sequence' in df.columns and 'HX time' in df.columns:
        key_counts = df.groupby(['Sequence', 'HX time']).size()
        duplicates = key_counts[key_counts > 1]
        if len(duplicates) > 0:
            dup_examples = [f"({seq}, {t})"
                            for (seq, t) in duplicates.index[:3]]
            n_dups = len(duplicates)
            errors.append(
                f"Duplicate (Sequence, HX time) keys found: "
                f"{n_dups} duplicated combinations "
                f"(e.g. {', '.join(dup_examples)})"
                f"{' ...' if n_dups > 3 else ''}"
            )

    # Check standard deviations are non-negative
    if 'Uptake SD (Da)' in df.columns:
        if (df['Uptake SD (Da)'] < 0).any():
            errors.append("'Uptake SD (Da)' cannot be negative")

    if errors:
        raise ValueError(
            "Aggregated data validation failed:\n  " + "\n  ".join(errors)
        )


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def calculate_peptide_length(sequence: str) -> int:
    """
    Calculate peptide length by counting only alphabetic characters.

    Parameters
    ----------
    sequence : str
        Peptide sequence string (may contain modifications/numbers)

    Returns
    -------
    int
        Number of amino acid letters in the sequence

    Examples
    --------
    >>> calculate_peptide_length("ASQFV")
    5
    >>> calculate_peptide_length("A[+16]SQFV")  # With modification
    5
    """
    return len(re.sub(r'[^a-zA-Z]', '', sequence))


# ==============================================================================
# DATA LOADER CLASS
# ==============================================================================

class DataLoader:
    """
    Handles loading and preprocessing of HDX-MS data files.

    This class provides methods to load null experiment data (with individual
    replicate rows) and aggregated data (averaged across replicates), as well
    as utilities for creating lookup tables and generating replicate pairings.

    Parameters
    ----------
    config : dict
        Configuration dictionary from config_parser.parse_config()
    verbose : bool, optional
        Print progress messages (default True)

    Attributes
    ----------
    config : dict
        Stored configuration dictionary
    verbose : bool
        Whether to print progress messages

    Notes
    -----
    Configuration access uses the config_parser module helpers to ensure
    consistent behavior across the codebase.
    """

    def __init__(self, config: Dict[str, Any], verbose: bool = True):
        """
        Initialize the DataLoader.

        Parameters
        ----------
        config : dict
            Configuration dictionary from parse_config()
        verbose : bool, optional
            Print progress messages (default True)
        """
        self.config = config
        self.verbose = verbose

    def _print(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to expected format.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            Dataframe with standardized column names
        """
        df = df.copy()
        df.columns = df.columns.str.strip()

        column_mapping = {}
        for standard_name, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in df.columns and alias != standard_name:
                    column_mapping[alias] = standard_name
                    break

        if column_mapping:
            df = df.rename(columns=column_mapping)

        return df

    def _validate_columns(self, df: pd.DataFrame, required_columns: List[str],
                          file_path: str) -> None:
        """
        Validate that required columns are present.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to check
        required_columns : list
            List of required column names
        file_path : str
            Path to file (for error messages)

        Raises
        ------
        ValueError
            If any required columns are missing
        """
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            available = list(df.columns)
            error_msg = (
                f"Missing columns in {os.path.basename(file_path)}:\n"
                f"  Required: {required_columns}\n"
                f"  Missing: {missing_cols}\n"
                f"  Available: {available}"
            )
            raise ValueError(error_msg)

    def load_csv(self, file_path: str, data_format: str = 'null_experiment') -> pd.DataFrame:
        """
        Load a CSV file and standardize its format.

        Parameters
        ----------
        file_path : str
            Path to the CSV file
        data_format : str
            Either 'null_experiment' or 'aggregated'

        Returns
        -------
        pd.DataFrame
            Loaded and standardized dataframe

        Raises
        ------
        FileNotFoundError
            If file does not exist
        ValueError
            If file cannot be decoded or is missing required columns
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try different encodings
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode file: {file_path}")

        # Standardize columns
        df = self._standardize_columns(df)

        # Validate based on format
        if data_format == 'null_experiment':
            self._validate_columns(df, NULL_EXPERIMENT_COLUMNS, file_path)
        else:  # aggregated
            self._validate_columns(df, AGGREGATED_COLUMNS, file_path)

        return df

    def load_null_experiment(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load null experiment data (individual replicate rows).

        Parameters
        ----------
        file_path : str
            Path to null experiment CSV file

        Returns
        -------
        tuple
            (dataframe, structure_dict) where structure_dict contains:
            - time_points: List of labeling times
            - peptides: List of peptide sequences
            - replicates: List of replicate numbers
            - n_peptides: Number of unique peptides
            - n_times: Number of labeling times
            - n_replicates: Number of replicates
            - n_rows: Total number of rows
        """
        df = self.load_csv(file_path, data_format='null_experiment')

        # Extract structure
        time_points = sorted(df['HX time'].unique().tolist())
        peptides = df['Sequence'].unique().tolist()
        replicates = sorted(df['Replicate number'].unique().tolist())

        structure = {
            'time_points': time_points,
            'peptides': peptides,
            'replicates': replicates,
            'n_peptides': len(peptides),
            'n_times': len(time_points),
            'n_replicates': len(replicates),
            'n_rows': len(df)
        }

        self._print(
            f"Loaded null experiment: {structure['n_peptides']} peptides, "
            f"{structure['n_times']} time points, {structure['n_replicates']} replicates"
        )

        return df, structure

    def load_aggregated_data(self, file_path: str) -> pd.DataFrame:
        """
        Load aggregated data (biosimilar or reference).

        Parameters
        ----------
        file_path : str
            Path to aggregated CSV file

        Returns
        -------
        pd.DataFrame
            Loaded dataframe with columns: Sequence, HX time,
            Uptake (Da), Uptake SD (Da), Replicate number
        """
        df = self.load_csv(file_path, data_format='aggregated')
        validate_aggregated_data(df)
        return df

    def create_uptake_lookup(self, df: pd.DataFrame) -> Dict[Tuple, float]:
        """
        Create a dictionary for O(1) lookup of uptake values.

        Parameters
        ----------
        df : pd.DataFrame
            Null experiment dataframe with individual replicate rows

        Returns
        -------
        dict
            Dictionary mapping (Sequence, Replicate, Time) -> Uptake value

        Notes
        -----
        This lookup table is used during precomputation to quickly access
        uptake values without repeated DataFrame filtering.
        """
        uptake_lookup = {}
        for _, row in df.iterrows():
            key = (row['Sequence'], row['Replicate number'], row['HX time'])
            uptake_lookup[key] = row['Uptake (Da)']
        return uptake_lookup

    def generate_all_pairings(self, replicates: List[int],
                               group_size: int) -> List[Dict]:
        """
        Generate all possible non-overlapping replicate pairings.

        For n replicates with group_size k, the number of pairings is:
        C(n, k) x C(n-k, k) / 2

        For 8 replicates with group_size=3: C(8,3) x C(5,3) / 2 = 280 pairings

        Parameters
        ----------
        replicates : list of int
            List of replicate IDs (e.g., [1, 2, 3, 4, 5, 6, 7, 8])
        group_size : int
            Number of replicates per group (e.g., 3)

        Returns
        -------
        list of dict
            Each dict contains:
            - pairing_id: Unique identifier (0-based)
            - ref_reps: List of replicate IDs for reference group
            - sample_reps: List of replicate IDs for sample group

        Raises
        ------
        ValueError
            If len(replicates) < 2 * group_size

        Examples
        --------
        >>> loader.generate_all_pairings([1, 2, 3, 4, 5, 6, 7, 8], 3)
        # Returns 280 pairings
        """
        if len(replicates) < 2 * group_size:
            raise ValueError(
                f"Need at least {2 * group_size} replicates for group_size={group_size}, "
                f"but only have {len(replicates)}"
            )

        all_groups = list(combinations(replicates, group_size))
        all_pairings = []
        pairing_id = 0

        for i in range(len(all_groups)):
            for j in range(i + 1, len(all_groups)):
                group_ref = all_groups[i]
                group_sample = all_groups[j]

                # Check if disjoint (non-overlapping)
                if len(set(group_ref) & set(group_sample)) == 0:
                    all_pairings.append({
                        'pairing_id': pairing_id,
                        'ref_reps': list(group_ref),
                        'sample_reps': list(group_sample)
                    })
                    pairing_id += 1

        self._print(f"Generated {len(all_pairings)} replicate pairings")
        return all_pairings


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    print("DATA LOADER MODULE - TEST")
    print("-" * 60)

    # Test 1: Pairing generation with different configurations
    print("\nTest 1: Pairing generation")

    # Create minimal config for testing
    test_config = {
        'STATISTICAL_PARAMETERS': {
            'group_size': '3',
            'n_replicates': '8'
        }
    }

    loader = DataLoader(test_config, verbose=True)

    # Test with standard parameters
    replicates = list(range(1, 9))  # [1, 2, 3, 4, 5, 6, 7, 8]
    pairings = loader.generate_all_pairings(replicates, group_size=3)
    print(f"  n_replicates=8, group_size=3 -> {len(pairings)} pairings (expected: 280)")

    # Test with different configurations
    for n_rep, g_size, expected in [(10, 3, 840), (8, 4, 35), (12, 4, 495)]:
        reps = list(range(1, n_rep + 1))
        pairs = loader.generate_all_pairings(reps, g_size)
        status = "OK" if len(pairs) == expected else "MISMATCH"
        print(f"  n_replicates={n_rep}, group_size={g_size} -> {len(pairs)} pairings (expected: {expected}) [{status}]")

    # Test 2: Peptide length calculation
    print("\nTest 2: Peptide length calculation")
    test_sequences = [
        ("ASQFV", 5),
        ("SVTEQDSKD", 9),
        ("A[+16]SQFV", 5),  # With modification marker
        ("IFPPSDEQLKSGTA", 14),
    ]

    for seq, expected in test_sequences:
        length = calculate_peptide_length(seq)
        status = "OK" if length == expected else "MISMATCH"
        print(f"  '{seq}' -> {length} (expected: {expected}) [{status}]")

    # Test 3: Validation functions (using mock data)
    print("\nTest 3: Validation functions")

    # Create mock null experiment data
    mock_null_df = pd.DataFrame({
        'Sequence': ['ASQFV'] * 48,  # 8 reps x 6 times
        'Replicate number': [r for r in range(1, 9)] * 6,
        'HX time': sorted([20, 100, 500, 2500, 12500, 62500] * 8),
        'Uptake (Da)': np.random.randn(48) + 1.0
    })

    try:
        validate_null_experiment_data(mock_null_df, expected_replicates=8)
        print("  Null experiment validation: PASSED")
    except ValueError as e:
        print(f"  Null experiment validation: FAILED - {e}")

    # Test with wrong number of replicates
    try:
        validate_null_experiment_data(mock_null_df, expected_replicates=10)
        print("  Mismatch validation: FAILED (should have raised error)")
    except ValueError:
        print("  Mismatch validation: PASSED (correctly raised error)")

    # Create mock aggregated data
    mock_agg_df = pd.DataFrame({
        'Sequence': ['ASQFV', 'SVTEQDSKD'],
        'HX time': [100, 100],
        'Uptake (Da)': [1.0, 2.0],
        'Uptake SD (Da)': [0.1, 0.2],
        'Replicate number': [3, 3]
    })

    try:
        validate_aggregated_data(mock_agg_df)
        print("  Aggregated data validation: PASSED")
    except ValueError as e:
        print(f"  Aggregated data validation: FAILED - {e}")

    # Test 4: File loading (if config exists)
    print("\nTest 4: File loading")
    from config_parser import parse_config

    config_path = "config.txt"
    if os.path.exists(config_path):
        config = parse_config(config_path)
        loader = DataLoader(config, verbose=True)

        sections = get_protein_sections(config)
        print(f"  Protein sections: {sections}")

        for section in sections[:1]:  # Test first section only
            info = get_candidates_from_section(config, section)
            print(f"\n  {section}:")
            print(f"    Display name: {info['display_name']}")
            print(f"    Null experiment: {os.path.basename(info['null_experiment'])}")
            print(f"    Reference: {os.path.basename(info['reference'])}")
            print(f"    Candidates: {len(info['candidates'])}")

            # Try to load the null experiment file
            null_path = info['null_experiment']
            if os.path.exists(null_path):
                df, structure = loader.load_null_experiment(null_path)
                print(f"\n    Loaded data structure:")
                print(f"      Peptides: {structure['n_peptides']}")
                print(f"      Time points: {structure['n_times']}")
                print(f"      Replicates: {structure['n_replicates']}")
                print(f"      Total rows: {structure['n_rows']}")

                # Validate against config
                n_rep = get_config_int(config, 'STATISTICAL_PARAMETERS', 'n_replicates', 8)
                try:
                    validate_null_experiment_data(df, n_rep)
                    print(f"      Validation against config (n_replicates={n_rep}): PASSED")
                except ValueError as e:
                    print(f"      Validation against config: FAILED - {e}")
    else:
        print(f"  Config file not found: {config_path}")
        print("  Skipping file loading tests.")

    print("\n" + "-" * 60)
    print("TEST COMPLETE")
