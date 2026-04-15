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
HDX-MS EQUIVALENCE TESTING - EXPERIMENT CONTEXT MODULE
================================================================================

PURPOSE:
    Provides the ExperimentContext class - a container for all experiment-derived
    values that are calculated once and passed to analysis modules. This eliminates
    redundant calculations and ensures consistency across the pipeline.

INPUTS:
    - Parsed configuration dictionary (from config_parser)
    - Data structure information (time points, peptides) set after loading data

OUTPUTS:
    - ExperimentContext object containing:
      - Config-derived values (alpha, group_size, n_replicates, etc.)
      - Calculated values (n_pairings, alpha_s, percentile, replicates list)
      - Data-derived values (time_points, peptides, n_times, n_peptides)

DEPENDENCIES:
    - config_parser.py (for config access helpers)
    - statistics.py (for calculate_sidak_alpha, calculate_n_pairings)

USAGE:
    from experiment import ExperimentContext
    from config_parser import parse_config, validate_config
    
    config = parse_config("config.txt")
    validate_config(config)
    
    context = ExperimentContext.from_config(config)
    
    # After loading data:
    context.set_data_structure(time_points=[20, 100, 500, 2500, 12500, 62500],
                                peptides=['ASQFV', 'SVTEQDSKD', ...])
    
    # Pass to analysis modules:
    analyzer = DirectPercentileAnalyzer(context, precomputed_df)

================================================================================
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from config_parser import (
    get_config_float,
    get_config_int,
    get_config_bool,
    get_config_value
)
from statistics_util import (
    calculate_sidak_alpha,
    calculate_n_pairings,
    calculate_percentile_from_alpha,
    SIDAK_M_PARAMETER
)


@dataclass
class ExperimentContext:
    """
    Container for all experiment parameters and derived values.
    
    This class serves as a single source of truth for experiment configuration,
    eliminating the need for modules to independently calculate values like
    n_pairings or alpha_s.
    
    Attributes
    ----------
    # From config (set at initialization)
    alpha : float
        Family-wise significance level (e.g., 0.05)
    alpha_ci : float
        Significance level for confidence intervals (e.g., 0.10 for 90% CI)
    group_size : int
        Number of replicates per group in pairings (e.g., 3)
    n_replicates : int
        Total number of replicates in null experiment (e.g., 8)
    n_simulations : int
        Number of Monte Carlo simulations to run
    
    # Calculated from config (set at initialization)
    n_pairings : int
        Number of non-overlapping replicate pairings (e.g., 280 for n=8, k=3)
    alpha_s : float
        Šidák-corrected significance level
    percentile : float
        Percentile corresponding to alpha_s (e.g., 97.47)
    replicates : List[int]
        List of replicate IDs [1, 2, ..., n_replicates]
    m : int
        Number of hypothesis tests (SIDAK_M_PARAMETER = 2)
    
    # From data (set after loading via set_data_structure)
    time_points : List[float]
        Sorted list of labeling times in seconds
    peptides : List[str]
        List of peptide sequences
    n_times : int
        Number of labeling time points
    n_peptides : int
        Number of peptides
    """
    
    # -------------------------------------------------------------------------
    # From config
    # -------------------------------------------------------------------------
    alpha: float
    alpha_ci: float
    group_size: int
    n_replicates: int
    n_simulations: int
    
    # -------------------------------------------------------------------------
    # Calculated from config
    # -------------------------------------------------------------------------
    n_pairings: int = field(init=False)
    alpha_s: float = field(init=False)
    percentile: float = field(init=False)
    replicates: List[int] = field(init=False)
    m: int = field(default=SIDAK_M_PARAMETER, init=False)
    
    # -------------------------------------------------------------------------
    # From data (set later via set_data_structure)
    # -------------------------------------------------------------------------
    time_points: List[float] = field(default_factory=list, init=False)
    peptides: List[str] = field(default_factory=list, init=False)
    n_times: int = field(default=0, init=False)
    n_peptides: int = field(default=0, init=False)
    
    # -------------------------------------------------------------------------
    # Optional config values
    # -------------------------------------------------------------------------
    random_seed: Optional[int] = None
    batch_size: int = 100000
    verbose: bool = True
    dD: Optional[float] = None
    EAC: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived values after initialization."""
        # Calculate number of pairings
        self.n_pairings = calculate_n_pairings(self.n_replicates, self.group_size)
        
        # Calculate Šidák-corrected alpha
        self.alpha_s = calculate_sidak_alpha(self.alpha, self.m)
        
        # Calculate corresponding percentile
        self.percentile = calculate_percentile_from_alpha(self.alpha_s)
        
        # Create list of replicate IDs
        self.replicates = list(range(1, self.n_replicates + 1))
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ExperimentContext':
        """
        Create ExperimentContext from parsed configuration dictionary.
        
        Parameters
        ----------
        config : dict
            Parsed configuration from config_parser.parse_config()
            
        Returns
        -------
        ExperimentContext
            Initialized context with all config-derived values
            
        Notes
        -----
        Call set_data_structure() after loading null experiment data
        to populate time_points, peptides, n_times, and n_peptides.
        """
        # Required parameters
        alpha = get_config_float(config, 'STATISTICAL_PARAMETERS', 'alpha', 0.05)
        alpha_ci = get_config_float(config, 'STATISTICAL_PARAMETERS', 'alpha_ci', 0.10)
        group_size = get_config_int(config, 'STATISTICAL_PARAMETERS', 'group_size', 3)
        n_replicates = get_config_int(config, 'STATISTICAL_PARAMETERS', 'n_replicates', 8)
        n_simulations = get_config_int(config, 'STATISTICAL_PARAMETERS', 'n_simulations', 100000000)
        
        # Optional parameters
        random_seed_str = get_config_value(config, 'ADVANCED', 'random_seed', '')
        random_seed = int(random_seed_str) if random_seed_str and random_seed_str.strip() else None
        
        batch_size = get_config_int(config, 'ADVANCED', 'batch_size', 100000)
        verbose = get_config_bool(config, 'ADVANCED', 'verbose', True)
        dD_str = get_config_value(config, 'STATISTICAL_PARAMETERS', 'dD', '')
        eac_str = get_config_value(config, 'STATISTICAL_PARAMETERS', 'EAC', '')
        dD = float(dD_str) if dD_str and dD_str.strip() else None
        eac = float(eac_str) if eac_str and eac_str.strip() else None
        
        return cls(
            alpha=alpha,
            alpha_ci=alpha_ci,
            group_size=group_size,
            n_replicates=n_replicates,
            n_simulations=n_simulations,
            random_seed=random_seed,
            batch_size=batch_size,
            verbose=verbose,
            dD=dD,
            EAC=eac
        )
    
    def set_data_structure(self, time_points: List[float], peptides: List[str]) -> None:
        """
        Set data-derived values after loading null experiment data.
        
        Parameters
        ----------
        time_points : list of float
            Labeling times in seconds (will be sorted)
        peptides : list of str
            Peptide sequences
            
        Notes
        -----
        This method should be called after loading the null experiment
        data, before passing the context to analysis modules.
        """
        self.time_points = sorted(time_points)
        self.peptides = list(peptides)
        self.n_times = len(self.time_points)
        self.n_peptides = len(self.peptides)
    
    def validate(self) -> None:
        """
        Validate that all required fields are set.
        
        Raises
        ------
        ValueError
            If data structure has not been set or values are invalid
        """
        errors = []
        
        # Check config-derived values
        if self.alpha <= 0 or self.alpha >= 1:
            errors.append(f"alpha must be between 0 and 1, got {self.alpha}")
        
        if self.alpha_ci <= 0 or self.alpha_ci >= 1:
            errors.append(f"alpha_ci must be between 0 and 1, got {self.alpha_ci}")
        
        if self.group_size < 2:
            errors.append(f"group_size must be at least 2, got {self.group_size}")
        
        if self.n_replicates < 2 * self.group_size:
            errors.append(
                f"n_replicates ({self.n_replicates}) must be at least "
                f"2 × group_size ({2 * self.group_size})"
            )
        
        if self.n_pairings <= 0:
            errors.append(f"n_pairings must be positive, got {self.n_pairings}")
        
        # Check data-derived values
        if self.n_times == 0:
            errors.append("time_points not set - call set_data_structure() first")
        
        if self.n_peptides == 0:
            errors.append("peptides not set - call set_data_structure() first")
        
        if errors:
            raise ValueError(
                "ExperimentContext validation failed:\n  " + "\n  ".join(errors)
            )
    
    def get_time_labels(self) -> List[str]:
        """
        Get LaTeX-formatted time labels for figures (t₁, t₂, etc.).
        
        Returns
        -------
        list of str
            Labels like ['$t_1$', '$t_2$', ...]
        """
        return [f'$t_{i+1}$' for i in range(self.n_times)]
    
    def summary(self) -> str:
        """
        Get a formatted summary string of the context.
        
        Returns
        -------
        str
            Multi-line summary of all parameters
        """
        lines = [
            "ExperimentContext Summary",
            "=" * 40,
            "",
            "Configuration Parameters:",
            f"  alpha = {self.alpha}",
            f"  alpha_ci = {self.alpha_ci}",
            f"  group_size = {self.group_size}",
            f"  n_replicates = {self.n_replicates}",
            f"  n_simulations = {self.n_simulations:,}",
            "",
            "Calculated Values:",
            f"  n_pairings = {self.n_pairings}",
            f"  alpha_s (Šidák) = {self.alpha_s:.6f}",
            f"  percentile = {self.percentile:.4f}th",
            f"  replicates = {self.replicates}",
            f"  m (tests) = {self.m}",
            "",
            "Data Structure:",
            f"  n_times = {self.n_times}",
            f"  n_peptides = {self.n_peptides}",
            f"  time_points = {self.time_points if self.time_points else '(not set)'}",
            "",
            "Optional Settings:",
            f"  random_seed = {self.random_seed}",
            f"  batch_size = {self.batch_size:,}",
            f"  verbose = {self.verbose}",
        ]
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Concise representation for debugging."""
        return (
            f"ExperimentContext(n_replicates={self.n_replicates}, "
            f"group_size={self.group_size}, n_pairings={self.n_pairings}, "
            f"n_times={self.n_times}, n_peptides={self.n_peptides})"
        )


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    print("EXPERIMENT CONTEXT MODULE - TEST")
    print("-" * 60)
    
    # Test 1: Manual creation
    print("\nTest 1: Manual creation")
    context = ExperimentContext(
        alpha=0.05,
        alpha_ci=0.10,
        group_size=3,
        n_replicates=8,
        n_simulations=1000000
    )
    
    print(f"  n_pairings = {context.n_pairings}")
    print(f"  alpha_s = {context.alpha_s:.6f}")
    print(f"  percentile = {context.percentile:.4f}th")
    print(f"  replicates = {context.replicates}")
    
    # Test 2: Set data structure
    print("\nTest 2: Set data structure")
    context.set_data_structure(
        time_points=[62500, 20, 100, 500, 2500, 12500],  # Unsorted to test sorting
        peptides=['ASQFV', 'SVTEQDSKD', 'IFPPSDEQLKSGTA']
    )
    
    print(f"  n_times = {context.n_times}")
    print(f"  n_peptides = {context.n_peptides}")
    print(f"  time_points (sorted) = {context.time_points}")
    print(f"  time_labels = {context.get_time_labels()}")
    
    # Test 3: Validation
    print("\nTest 3: Validation")
    try:
        context.validate()
        print("  Validation: PASSED")
    except ValueError as e:
        print(f"  Validation FAILED: {e}")
    
    # Test 4: From config (simulated)
    print("\nTest 4: From config (simulated)")
    fake_config = {
        'STATISTICAL_PARAMETERS': {
            'alpha': '0.05',
            'alpha_ci': '0.10',
            'group_size': '3',
            'n_replicates': '8',
            'n_simulations': '100000000'
        },
        'ADVANCED': {
            'random_seed': '',
            'batch_size': '100000',
            'verbose': 'true'
        }
    }
    
    context2 = ExperimentContext.from_config(fake_config)
    print(f"  Created from config: {context2}")
    
    # Test 5: Different configurations
    print("\nTest 5: Different configurations")
    for n_rep, g_size in [(8, 3), (10, 3), (8, 4), (12, 4)]:
        ctx = ExperimentContext(
            alpha=0.05, alpha_ci=0.10,
            group_size=g_size, n_replicates=n_rep,
            n_simulations=1000
        )
        print(f"  n_rep={n_rep}, group_size={g_size} → n_pairings={ctx.n_pairings}")
    
    # Test 6: Summary output
    print("\nTest 6: Summary")
    print(context.summary())
    
    print("\n" + "-" * 60)
    print("TEST COMPLETE")
