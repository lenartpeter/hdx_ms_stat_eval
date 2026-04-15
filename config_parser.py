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
HDX-MS EQUIVALENCE TESTING - CONFIG PARSER MODULE
================================================================================

PURPOSE:
    Configuration file parsing, validation, and
    access helper functions. All other modules import config utilities from
    here.

INPUTS:
    - Path to INI-style configuration file (config.txt)

OUTPUTS:
    - Parsed configuration dictionary
    - Validated configuration (raises ConfigError if invalid)
    - Helper functions for type-safe config value access

DEPENDENCIES:
    - None (standard library only)

USAGE:
    from config_parser import parse_config, validate_config, get_config_float
    
    config = parse_config("config.txt")
    validate_config(config)
    alpha = get_config_float(config, 'STATISTICAL_PARAMETERS', 'alpha', 0.05)

================================================================================
"""

import os
from typing import Dict, List, Any, Optional


# ==============================================================================
# EXCEPTIONS
# ==============================================================================

class ConfigError(Exception):
    """
    Raised when configuration validation fails.
    
    Contains a descriptive message listing all validation errors found.
    """
    pass


# ==============================================================================
# CONFIG PARSER
# ==============================================================================

def parse_config(config_path: str) -> Dict[str, Dict[str, str]]:
    """
    Parse an INI-style configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file
        
    Returns
    -------
    dict
        Nested dictionary: {section_name: {key: value}}
        
    Raises
    ------
    FileNotFoundError
        If config file does not exist
        
    Notes
    -----
    - Lines starting with # are treated as comments
    - Inline comments (# after value) are stripped
    - Keys and values are stripped of whitespace
    - Empty values are preserved as empty strings
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = {}
    current_section = None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Section header
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].strip()
                config[current_section] = {}
                continue
            
            # Key-value pair
            if '=' in line and current_section is not None:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle inline comments (but preserve # in file paths)
                # Only strip comment if there's a space before #
                if ' #' in value:
                    value = value.split(' #')[0].strip()
                
                config[current_section][key] = value
            elif '=' in line and current_section is None:
                pass
    
    return config


# ==============================================================================
# CONFIG VALIDATION
# ==============================================================================

def validate_config(config: Dict[str, Dict[str, str]]) -> None:
    """
    Validate configuration dictionary structure and required values.
    
    Parameters
    ----------
    config : dict
        Parsed configuration dictionary from parse_config()
        
    Raises
    ------
    ConfigError
        If any validation check fails. The error message lists all
        problems found, not just the first one.
        
    Notes
    -----
    Checks performed:
    - Required sections exist
    - Required keys in STATISTICAL_PARAMETERS
    - At least one PROTEIN_TO_EVALUATE_N section
    - Each protein section has required keys and at least one candidate
    - File paths exist (warnings only, not errors)
    """
    errors = []
    warnings = []
    
    # -------------------------------------------------------------------------
    # Check required sections
    # -------------------------------------------------------------------------
    required_sections = [
        'STATISTICAL_PARAMETERS',
        'ANALYSIS_OPTIONS', 
        'OUTPUT_SETTINGS',
        'PRECOMPUTATION'
    ]
    
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: [{section}]")
    
    # -------------------------------------------------------------------------
    # Check STATISTICAL_PARAMETERS keys
    # -------------------------------------------------------------------------
    if 'STATISTICAL_PARAMETERS' in config:
        stat_params = config['STATISTICAL_PARAMETERS']
        required_keys = [
            'alpha',
            'alpha_ci',
            'alpha_significance_test',
            'group_size',
            'n_replicates'
        ]
        
        for key in required_keys:
            if key not in stat_params:
                errors.append(f"Missing required key in [STATISTICAL_PARAMETERS]: {key}")
            elif not stat_params[key].strip():
                errors.append(f"Empty value for [STATISTICAL_PARAMETERS] {key}")
        
        # Validate numeric ranges if keys exist
        if 'alpha' in stat_params and stat_params['alpha'].strip():
            try:
                alpha = float(stat_params['alpha'])
                if not (0 < alpha < 1):
                    errors.append(f"alpha must be between 0 and 1, got {alpha}")
            except ValueError:
                errors.append(f"alpha must be numeric, got '{stat_params['alpha']}'")
        
        if 'alpha_ci' in stat_params and stat_params['alpha_ci'].strip():
            try:
                alpha_ci = float(stat_params['alpha_ci'])
                if not (0 < alpha_ci < 1):
                    errors.append(f"alpha_ci must be between 0 and 1, got {alpha_ci}")
            except ValueError:
                errors.append(f"alpha_ci must be numeric, got '{stat_params['alpha_ci']}'")

        if ('alpha_significance_test' in stat_params
                and stat_params['alpha_significance_test'].strip()):
            try:
                alpha_significance_test = float(
                    stat_params['alpha_significance_test']
                )
                if not (0 < alpha_significance_test < 1):
                    errors.append(
                        "alpha_significance_test must be between 0 and 1, "
                        f"got {alpha_significance_test}"
                    )
            except ValueError:
                errors.append(
                    "alpha_significance_test must be numeric, got "
                    f"'{stat_params['alpha_significance_test']}'"
                )
        
        if 'group_size' in stat_params and stat_params['group_size'].strip():
            try:
                group_size = int(stat_params['group_size'])
                if group_size < 2:
                    errors.append(f"group_size must be at least 2, got {group_size}")
            except ValueError:
                errors.append(f"group_size must be integer, got '{stat_params['group_size']}'")
        
        if 'n_replicates' in stat_params and stat_params['n_replicates'].strip():
            try:
                n_replicates = int(stat_params['n_replicates'])
                if n_replicates < 4:
                    errors.append(f"n_replicates must be at least 4, got {n_replicates}")
            except ValueError:
                errors.append(f"n_replicates must be integer, got '{stat_params['n_replicates']}'")

        for optional_limit_key in ['dD', 'EAC']:
            if optional_limit_key in stat_params and stat_params[optional_limit_key].strip():
                try:
                    limit_value = float(stat_params[optional_limit_key])
                    if limit_value < 0:
                        errors.append(
                            f"{optional_limit_key} must be non-negative, got {limit_value}"
                        )
                except ValueError:
                    errors.append(
                        f"{optional_limit_key} must be numeric, got "
                        f"'{stat_params[optional_limit_key]}'"
                    )
        
        # Cross-validation: n_replicates must be >= 2 * group_size
        if ('group_size' in stat_params and 'n_replicates' in stat_params and
            stat_params['group_size'].strip() and stat_params['n_replicates'].strip()):
            try:
                group_size = int(stat_params['group_size'])
                n_replicates = int(stat_params['n_replicates'])
                if n_replicates < 2 * group_size:
                    errors.append(
                        f"n_replicates ({n_replicates}) must be at least "
                        f"2 * group_size ({2 * group_size}) for non-overlapping pairings"
                    )
            except ValueError:
                pass

    # -------------------------------------------------------------------------
    # Check PROTEIN_TO_EVALUATE sections
    # -------------------------------------------------------------------------
    protein_sections = [s for s in config if s.startswith('PROTEIN_TO_EVALUATE_')]
    
    if not protein_sections:
        errors.append("No [PROTEIN_TO_EVALUATE_N] sections found")
    
    for section in protein_sections:
        section_config = config[section]
        
        # Required keys
        for required in ['null_experiment', 'reference']:
            if required not in section_config:
                errors.append(f"[{section}] missing required key: {required}")
            elif not section_config[required].strip():
                errors.append(f"[{section}] {required} is empty")
            else:
                # Check file exists (warning only)
                path = section_config[required].strip()
                if not os.path.exists(path):
                    warnings.append(f"[{section}] {required} file not found: {path}")
        
        # At least one candidate
        candidates = [k for k in section_config if k.startswith('candidate_')]
        if not candidates:
            errors.append(f"[{section}] has no candidate_N entries")
        else:
            # Check candidate files exist 
            for cand_key in candidates:
                path = section_config[cand_key].strip()
                if path and not os.path.exists(path):
                    warnings.append(f"[{section}] {cand_key} file not found: {path}")
    
    # -------------------------------------------------------------------------
    # Report results
    # -------------------------------------------------------------------------
    if warnings:
        print("Config validation warnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")
    
    if errors:
        raise ConfigError(
            "Config validation failed:\n  " + "\n  ".join(errors)
        )


# ==============================================================================
# CONFIG ACCESS HELPERS
# ==============================================================================

def get_config_value(config: Dict, section: str, key: str, 
                     default: Any = None) -> Optional[str]:
    """
    Safely get a string value from the config dictionary.
    
    Parameters
    ----------
    config : dict
        Parsed configuration dictionary
    section : str
        Section name (e.g., 'STATISTICAL_PARAMETERS')
    key : str
        Key name within section
    default : any, optional
        Default value if key not found (default None)
        
    Returns
    -------
    str or None
        The config value as string, or default if not found
    """
    return config.get(section, {}).get(key, default)


def get_config_bool(config: Dict, section: str, key: str, 
                    default: bool = False) -> bool:
    """
    Get a boolean value from the config.
    
    Parameters
    ----------
    config : dict
        Parsed configuration dictionary
    section : str
        Section name
    key : str
        Key name
    default : bool, optional
        Default value if key not found (default False)
        
    Returns
    -------
    bool
        True if value is 'true', 'yes', '1', or 'on' (case-insensitive)
    """
    value = get_config_value(config, section, key, None)
    if value is None:
        return default
    return str(value).lower() in ('true', 'yes', '1', 'on')


def get_config_float(config: Dict, section: str, key: str, 
                     default: float = 0.0) -> float:
    """
    Get a float value from the config.
    
    Parameters
    ----------
    config : dict
        Parsed configuration dictionary
    section : str
        Section name
    key : str
        Key name
    default : float, optional
        Default value if key not found or invalid (default 0.0)
        
    Returns
    -------
    float
        Parsed float value, or default if parsing fails
    """
    value = get_config_value(config, section, key, None)
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def get_config_int(config: Dict, section: str, key: str, 
                   default: int = 0) -> int:
    """
    Get an integer value from the config.
    
    Parameters
    ----------
    config : dict
        Parsed configuration dictionary
    section : str
        Section name
    key : str
        Key name
    default : int, optional
        Default value if key not found or invalid (default 0)
        
    Returns
    -------
    int
        Parsed integer value, or default if parsing fails
        
    Notes
    -----
    Handles float strings like '100.0' by converting to float first.
    """
    value = get_config_value(config, section, key, None)
    if value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def get_config_list(config: Dict, section: str, key: str, 
                    default: Optional[List] = None) -> List[str]:
    """
    Get a list value from the config (comma-separated values).
    
    Parameters
    ----------
    config : dict
        Parsed configuration dictionary
    section : str
        Section name
    key : str
        Key name
    default : list, optional
        Default value if key not found (default empty list)
        
    Returns
    -------
    list
        List of stripped string values
        
    Example
    -------
    Config: "values = 6, 7, 8, 9"
    Returns: ['6', '7', '8', '9']
    """
    if default is None:
        default = []
    
    value = get_config_value(config, section, key, '')
    if not value:
        return default
    
    return [item.strip() for item in value.split(',') if item.strip()]


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_protein_sections(config: Dict) -> List[str]:
    """
    Get all PROTEIN_TO_EVALUATE_N section names from config.
    
    Parameters
    ----------
    config : dict
        Parsed configuration dictionary
        
    Returns
    -------
    list
        Sorted list of section names like ['PROTEIN_TO_EVALUATE_1', 'PROTEIN_TO_EVALUATE_2']
    """
    sections = [s for s in config if s.startswith('PROTEIN_TO_EVALUATE_')]
    return sorted(sections, key=lambda x: int(x.split('_')[-1]))


def get_candidates_from_section(config: Dict, section_name: str) -> Dict[str, Any]:
    """
    Extract all candidate information from a protein section.
    
    Parameters
    ----------
    config : dict
        Parsed configuration dictionary
    section_name : str
        Name of the protein section (e.g., 'PROTEIN_TO_EVALUATE_1')
        
    Returns
    -------
    dict
        {
            'display_name': str,
            'null_experiment': str (path),
            'reference': str (path),
            'candidates': {1: path, 2: path, ...}
        }
    """
    section = config.get(section_name, {})
    
    result = {
        'display_name': section.get('display_name', section_name).strip(),
        'null_experiment': section.get('null_experiment', '').strip(),
        'reference': section.get('reference', '').strip(),
        'candidates': {}
    }
    
    # Find candidates (candidate_1, candidate_2, etc.)
    n = 1
    while True:
        key = f'candidate_{n}'
        if key in section and section[key].strip():
            result['candidates'][n] = section[key].strip()
            n += 1
        else:
            break
    
    return result


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    print("CONFIG PARSER MODULE - TEST")
    print("-" * 60)
    
    # Test with actual config file if present
    config_path = "config.txt"
    
    if os.path.exists(config_path):
        print(f"Testing with: {config_path}")
        
        # Parse
        config = parse_config(config_path)
        print(f"\nSections found: {list(config.keys())}")
        
        # Validate
        try:
            validate_config(config)
            print("\nValidation: PASSED")
        except ConfigError as e:
            print(f"\nValidation FAILED:\n{e}")
        
        # Test helpers
        print("\nHelper function tests:")
        print(f"  alpha = {get_config_float(config, 'STATISTICAL_PARAMETERS', 'alpha', 0.05)}")
        print(f"  group_size = {get_config_int(config, 'STATISTICAL_PARAMETERS', 'group_size', 3)}")
        print(f"  run_monte_carlo = {get_config_bool(config, 'ANALYSIS_OPTIONS', 'run_monte_carlo', False)}")
        print(f"  se_n_partitions = {get_config_list(config, 'PARTITIONED_LIMITS', 'se_n_partitions', [])}")
        
        # Test protein sections
        protein_sections = get_protein_sections(config)
        print(f"\nProtein sections: {protein_sections}")
        
        for section in protein_sections:
            info = get_candidates_from_section(config, section)
            print(f"\n  {section}:")
            print(f"    Display name: {info['display_name']}")
            print(f"    Candidates: {len(info['candidates'])}")
    else:
        print(f"Config file not found: {config_path}")
        print("Create a config.txt file to test parsing.")
    
    print("\n" + "-" * 60)
    print("TEST COMPLETE")
