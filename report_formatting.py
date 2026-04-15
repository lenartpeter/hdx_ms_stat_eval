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
Shared helpers for report-only numeric formatting.
"""

from typing import Any, Dict

from config_parser import get_config_bool


DEFAULT_LIMIT_DECIMALS = 3
UNROUNDED_LIMIT_DECIMALS = 4


def should_round_reported_limits(config: Dict[str, Any]) -> bool:
    """
    Return whether final reported EAC and |ΔD|_limit values should be rounded.

    This affects presentation only. Calculations should continue to use the
    underlying unrounded values.
    """
    return get_config_bool(config or {}, 'ADVANCED', 'round_reported_limits', True)


def get_report_limit_decimals(config: Dict[str, Any]) -> int:
    """Get decimal places for final reported limits."""
    if should_round_reported_limits(config):
        return DEFAULT_LIMIT_DECIMALS
    return UNROUNDED_LIMIT_DECIMALS


def format_report_limit(value: float, config: Dict[str, Any]) -> str:
    """Format a final reported limit value for terminal/PDF/figure output."""
    decimals = get_report_limit_decimals(config)
    return f"{value:.{decimals}f}"
