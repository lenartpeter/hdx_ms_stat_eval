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
HDX-MS EQUIVALENCE TESTING - CENTRALIZED PLOTTING MODULE
================================================================================

PURPOSE:
    Centralized configuration for figures. Ensures consistent
    sizing, fonts, and styling across all plot types.

SUPPORTED FORMATS:
    - TIFF (default): Lossless raster format, widely accepted by journals
    - PNG: Compressed raster format
    - PDF: Vector format, infinitely scalable

FIGURE SIZES (in inches):
    - Direct Percentile: 5.7 x 2.46 (for ranked value plots)
    - Resampling: 6.5 x 1.8 (for delta D rank plots)
    - Uptake Range: 5.82 x 3.01 

CONFIG OPTIONS:
    [OUTPUT_SETTINGS]
    figure_format = tiff  # Options: tiff, png, pdf
    generate_figures = true

================================================================================
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
import os
from datetime import datetime

from config_parser import get_config_float, get_config_int, get_config_bool
from report_formatting import format_report_limit


# ==============================================================================
# SUPPORTED OUTPUT FORMATS
# ==============================================================================

SUPPORTED_FORMATS = ['tiff', 'png', 'pdf']
DEFAULT_FORMAT = 'tiff'


# ==============================================================================
# FIGURE SIZE CONSTANTS (defaults in inches, can be overridden by config)
# ==============================================================================

CM_TO_INCH = 1 / 2.54

# Direct Percentile method figures (ranked values plots)
DEFAULT_DIRECT_PERCENTILE_WIDTH_INCH = 5.7
DEFAULT_DIRECT_PERCENTILE_HEIGHT_INCH = 2.46
DIRECT_PERCENTILE_FIGSIZE = (DEFAULT_DIRECT_PERCENTILE_WIDTH_INCH,
                              DEFAULT_DIRECT_PERCENTILE_HEIGHT_INCH)

# Resampling method figures
DEFAULT_RESAMPLING_WIDTH_INCH = 11.4
DEFAULT_RESAMPLING_HEIGHT_INCH = 3.0
RESAMPLING_FIGSIZE = (DEFAULT_RESAMPLING_WIDTH_INCH,
                       DEFAULT_RESAMPLING_HEIGHT_INCH)

# Uptake Range plot (12.69 x 6.37 cm = 4.996 x 2.508 inches)
DEFAULT_UPTAKE_RANGE_WIDTH_INCH = 4.996
DEFAULT_UPTAKE_RANGE_HEIGHT_INCH = 2.508
UPTAKE_RANGE_FIGSIZE = (DEFAULT_UPTAKE_RANGE_WIDTH_INCH,
                         DEFAULT_UPTAKE_RANGE_HEIGHT_INCH)

# Default font sizes (points)
DEFAULT_UPTAKE_RANGE_AXIS_TITLE_FONTSIZE = 8.4
DEFAULT_UPTAKE_RANGE_AXIS_NUMBER_FONTSIZE = 7.2
DEFAULT_UPTAKE_RANGE_DOT_SIZE = 3

DEFAULT_DIRECT_PERCENTILE_AXIS_TITLE_FONTSIZE = 8.05
DEFAULT_DIRECT_PERCENTILE_AXIS_NUMBER_FONTSIZE = 7

DEFAULT_RESAMPLING_AXIS_TITLE_FONTSIZE = 12
DEFAULT_RESAMPLING_AXIS_NUMBER_FONTSIZE = 10
DEFAULT_RESAMPLING_RESULT_TEXT_FONTSIZE = 11
DEFAULT_RESAMPLING_DOT_SIZE = 3
DEFAULT_RESAMPLING_CI_LINE_WIDTH = 0.5
DEFAULT_RESAMPLING_LIMIT_LINE_WIDTH = 1.5

# DPI settings (defaults - can be overridden in config.txt)
DEFAULT_DPI = 300  # Default DPI for all figure types (sufficient for most journals)


# ==============================================================================
# FIGURE SETTINGS FROM CONFIG
# ==============================================================================

def get_figure_settings(config: Dict[str, Any], figure_type: str) -> Dict[str, Any]:
    """
    Get figure settings from config, with defaults if not specified.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    figure_type : str
        'uptake_range', 'direct_percentile', or 'resampling'

    Returns
    -------
    dict
        Dictionary with figsize, axis_title_fontsize, axis_number_fontsize, exact_figsize,
        and additional settings depending on figure type
    """
    # Get exact_figsize setting (applies to all figure types)
    exact_figsize = get_config_bool(config, 'FIGURE_SETTINGS', 'exact_figsize', True)

    if figure_type == 'uptake_range':
        width = get_config_float(config, 'FIGURE_SETTINGS', 'uptake_range_width_inch',
                                  DEFAULT_UPTAKE_RANGE_WIDTH_INCH)
        height = get_config_float(config, 'FIGURE_SETTINGS', 'uptake_range_height_inch',
                                   DEFAULT_UPTAKE_RANGE_HEIGHT_INCH)
        axis_title_fontsize = get_config_float(config, 'FIGURE_SETTINGS', 'uptake_range_axis_title_fontsize',
                                                DEFAULT_UPTAKE_RANGE_AXIS_TITLE_FONTSIZE)
        axis_number_fontsize = get_config_float(config, 'FIGURE_SETTINGS', 'uptake_range_axis_number_fontsize',
                                                 DEFAULT_UPTAKE_RANGE_AXIS_NUMBER_FONTSIZE)
        dot_size = get_config_float(config, 'FIGURE_SETTINGS', 'uptake_range_dot_size',
                                     DEFAULT_UPTAKE_RANGE_DOT_SIZE)
        dpi = get_config_int(config, 'FIGURE_SETTINGS', 'uptake_range_dpi', DEFAULT_DPI)
        return {
            'figsize': (width, height),
            'axis_title_fontsize': axis_title_fontsize,
            'axis_number_fontsize': axis_number_fontsize,
            'dot_size': dot_size,
            'exact_figsize': exact_figsize,
            'dpi': dpi
        }

    elif figure_type == 'direct_percentile':
        width = get_config_float(config, 'FIGURE_SETTINGS', 'direct_percentile_width_inch',
                                  DEFAULT_DIRECT_PERCENTILE_WIDTH_INCH)
        height = get_config_float(config, 'FIGURE_SETTINGS', 'direct_percentile_height_inch',
                                   DEFAULT_DIRECT_PERCENTILE_HEIGHT_INCH)
        axis_title_fontsize = get_config_float(config, 'FIGURE_SETTINGS', 'direct_percentile_axis_title_fontsize',
                                                DEFAULT_DIRECT_PERCENTILE_AXIS_TITLE_FONTSIZE)
        axis_number_fontsize = get_config_float(config, 'FIGURE_SETTINGS', 'direct_percentile_axis_number_fontsize',
                                                 DEFAULT_DIRECT_PERCENTILE_AXIS_NUMBER_FONTSIZE)
        dpi = get_config_int(config, 'FIGURE_SETTINGS', 'direct_percentile_dpi', DEFAULT_DPI)
        return {
            'figsize': (width, height),
            'axis_title_fontsize': axis_title_fontsize,
            'axis_number_fontsize': axis_number_fontsize,
            'exact_figsize': exact_figsize,
            'dpi': dpi
        }

    elif figure_type == 'resampling':
        width = get_config_float(config, 'FIGURE_SETTINGS', 'resampling_width_inch',
                                  DEFAULT_RESAMPLING_WIDTH_INCH)
        height = get_config_float(config, 'FIGURE_SETTINGS', 'resampling_height_inch',
                                   DEFAULT_RESAMPLING_HEIGHT_INCH)
        axis_title_fontsize = get_config_float(config, 'FIGURE_SETTINGS', 'resampling_axis_title_fontsize',
                                                DEFAULT_RESAMPLING_AXIS_TITLE_FONTSIZE)
        axis_number_fontsize = get_config_float(config, 'FIGURE_SETTINGS', 'resampling_axis_number_fontsize',
                                                 DEFAULT_RESAMPLING_AXIS_NUMBER_FONTSIZE)
        result_text_fontsize = get_config_float(config, 'FIGURE_SETTINGS', 'resampling_result_text_fontsize',
                                                 DEFAULT_RESAMPLING_RESULT_TEXT_FONTSIZE)
        dot_size = get_config_float(config, 'FIGURE_SETTINGS', 'resampling_dot_size',
                                     DEFAULT_RESAMPLING_DOT_SIZE)
        ci_line_width = get_config_float(config, 'FIGURE_SETTINGS', 'resampling_ci_line_width',
                                          DEFAULT_RESAMPLING_CI_LINE_WIDTH)
        limit_line_width = get_config_float(config, 'FIGURE_SETTINGS', 'resampling_limit_line_width',
                                             DEFAULT_RESAMPLING_LIMIT_LINE_WIDTH)
        dpi = get_config_int(config, 'FIGURE_SETTINGS', 'resampling_dpi', DEFAULT_DPI)
        return {
            'figsize': (width, height),
            'axis_title_fontsize': axis_title_fontsize,
            'axis_number_fontsize': axis_number_fontsize,
            'result_text_fontsize': result_text_fontsize,
            'dot_size': dot_size,
            'ci_line_width': ci_line_width,
            'limit_line_width': limit_line_width,
            'exact_figsize': exact_figsize,
            'dpi': dpi
        }

    elif figure_type == 'volcano':
        return {
            'figsize': (
                get_config_float(config, 'FIGURE_SETTINGS',
                                 'volcano_width_inch', 3.7),
                get_config_float(config, 'FIGURE_SETTINGS',
                                 'volcano_height_inch', 4.0)
            ),
            'axis_title_fontsize': get_config_float(
                config, 'FIGURE_SETTINGS',
                'volcano_axis_title_fontsize', 8),
            'axis_number_fontsize': get_config_float(
                config, 'FIGURE_SETTINGS',
                'volcano_axis_number_fontsize', 7),
            'dot_size': get_config_float(
                config, 'FIGURE_SETTINGS',
                'volcano_dot_size', 8),
            'exact_figsize': exact_figsize,
            'dpi': get_config_int(
                config, 'FIGURE_SETTINGS',
                'volcano_dpi', DEFAULT_DPI)
        }

    else:
        # Default fallback
        return {
            'figsize': DIRECT_PERCENTILE_FIGSIZE,
            'axis_title_fontsize': 7,
            'axis_number_fontsize': 7,
            'exact_figsize': exact_figsize,
            'dpi': DEFAULT_DPI
        }


# ==============================================================================
# COLOR PALETTE
# ==============================================================================

# Colors for time points (matplotlib defaults, more saturated)
FIGURE_COLORS = [
    '#1f77b4',  # t1: Blue
    '#ff7f0e',  # t2: Orange
    '#2ca02c',  # t3: Green
    '#d62728',  # t4: Red
    '#9467bd',  # t5: Purple
    '#8c564b',  # t6: Brown
    '#e377c2',  # t7: Pink
    '#7f7f7f',  # t8: Gray
    '#bcbd22',  # t9: Olive
    '#17becf',  # t10: Cyan
]


# ==============================================================================
# FORMAT UTILITIES
# ==============================================================================

def get_format_extension(output_format: str) -> str:
    """Get file extension for the given format."""
    fmt = output_format.lower() if output_format else DEFAULT_FORMAT
    if fmt not in SUPPORTED_FORMATS:
        fmt = DEFAULT_FORMAT
    return f'.{fmt}'


def validate_format(output_format: str) -> str:
    """Validate and normalize output format string."""
    if output_format is None:
        return DEFAULT_FORMAT
    fmt = output_format.lower().strip()
    if fmt not in SUPPORTED_FORMATS:
        print(f"Warning: Unknown format '{output_format}', using '{DEFAULT_FORMAT}'")
        return DEFAULT_FORMAT
    return fmt


# ==============================================================================
# MATPLOTLIB CONFIGURATION
# ==============================================================================

def configure_matplotlib_publication() -> None:

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    mpl.rcParams['font.size'] = 7
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['lines.linewidth'] = 0.5
    mpl.rcParams['xtick.major.width'] = 0.5
    mpl.rcParams['ytick.major.width'] = 0.5
    mpl.rcParams['xtick.major.size'] = 2
    mpl.rcParams['ytick.major.size'] = 2
    # For PDF/EPS font embedding
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42


def configure_matplotlib_direct_percentile() -> None:

    configure_matplotlib_publication()
    mpl.rcParams['xtick.major.size'] = 1
    mpl.rcParams['ytick.major.size'] = 1


def configure_matplotlib_uptake_range() -> None:

    configure_matplotlib_publication()
    mpl.rcParams['font.size'] = 7  # axis labels
    # Tick labels will be set to 6 pt in the plot function


# ==============================================================================
# FIGURE CREATION
# ==============================================================================

def create_figure(figsize: Tuple[float, float] = None,
                  figure_type: str = 'direct_percentile') -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a figure with consistent sizing.

    Parameters
    ----------
    figsize : tuple, optional
        Custom figure size (width, height) in inches.
        If None, uses default for figure_type.
    figure_type : str
        'direct_percentile', 'resampling', or 'uptake_range'

    Returns
    -------
    tuple
        (fig, ax)
    """
    if figsize is None:
        if figure_type == 'direct_percentile':
            figsize = DIRECT_PERCENTILE_FIGSIZE
        elif figure_type == 'resampling':
            figsize = RESAMPLING_FIGSIZE
        elif figure_type == 'volcano':
            figsize = (3.7, 4.0)
        elif figure_type == 'uptake_range':
            figsize = UPTAKE_RANGE_FIGSIZE
        else:
            figsize = DIRECT_PERCENTILE_FIGSIZE

    if figure_type == 'direct_percentile':
        configure_matplotlib_direct_percentile()
    elif figure_type == 'uptake_range':
        configure_matplotlib_uptake_range()
    else:
        configure_matplotlib_publication()

    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ==============================================================================
# FIGURE SAVING
# ==============================================================================

def save_figure(fig: plt.Figure,
                output_path: str,
                output_format: str = None,
                dpi: int = None,
                figure_type: str = 'direct_percentile',
                exact_size: bool = True) -> str:
    """
    Save figure with consistent settings.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save
    output_path : str
        Output file path (extension will be replaced based on output_format)
    output_format : str, optional
        Output format: 'tiff' (default), 'png', or 'pdf'
    dpi : int, optional
        DPI for raster formats. If None, uses DEFAULT_DPI (300).
    figure_type : str
        'direct_percentile', 'resampling', or 'uptake_range'
    exact_size : bool
        If True, save at exact figsize dimensions (no tight bbox cropping).
        If False, use tight bounding box (may crop whitespace).

    Returns
    -------
    str
        Actual path where figure was saved
    """
    # Validate format
    fmt = validate_format(output_format)

    # Set DPI (use default if not provided - config DPI should be passed explicitly)
    if dpi is None:
        dpi = DEFAULT_DPI

    # Update path with correct extension
    base_path = os.path.splitext(output_path)[0]
    actual_path = base_path + get_format_extension(fmt)

    # Use LZW compression for TIFF (lossless, ~2-4x smaller files)
    pil_kwargs = {'compression': 'tiff_lzw'} if fmt == 'tiff' else {}

    # Save figure
    if exact_size:
        # Save at exact specified dimensions (no bbox adjustment)
        fig.savefig(
            actual_path,
            dpi=dpi,
            facecolor='white',
            edgecolor='none',
            format=fmt,
            pil_kwargs=pil_kwargs
        )
    else:
        # Use tight bounding box (may change dimensions)
        fig.savefig(
            actual_path,
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0.02,
            facecolor='white',
            edgecolor='none',
            format=fmt,
            pil_kwargs=pil_kwargs
        )

    return actual_path


def save_figure_exact(fig: plt.Figure,
                      output_path: str,
                      output_format: str = None,
                      dpi: int = None,
                      figure_type: str = 'direct_percentile') -> str:
    """
    Saves figure with exact pixel dimensions (no tight bounding box adjustment).
    The figure will be saved at exactly figsize * dpi pixels.

    Note: This may include extra whitespace if the plot doesn't fill the figure.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save
    output_path : str
        Output file path
    output_format : str, optional
        Output format: 'tiff' (default), 'png', or 'pdf'
    dpi : int, optional
        DPI for output
    figure_type : str
        'direct_percentile', 'resampling', or 'uptake_range'

    Returns
    -------
    str
        Path where figure was saved
    """
    fmt = validate_format(output_format)

    if dpi is None:
        dpi = DEFAULT_DPI

    base_path = os.path.splitext(output_path)[0]
    actual_path = base_path + get_format_extension(fmt)

    # Use LZW compression for TIFF (lossless, ~2-4x smaller files)
    pil_kwargs = {'compression': 'tiff_lzw'} if fmt == 'tiff' else {}

    # Save WITHOUT bbox_inches='tight' for exact dimensions
    fig.savefig(
        actual_path,
        dpi=dpi,
        facecolor='white',
        edgecolor='none',
        format=fmt,
        pil_kwargs=pil_kwargs
    )

    return actual_path


# ==============================================================================
# STYLING HELPERS
# ==============================================================================

def style_axes_publication(ax: plt.Axes, hide_top_right: bool = True) -> None:
    """
    Parameters
    ----------
    ax : Axes
        Matplotlib axes to style
    hide_top_right : bool
        If True, hides top and right spines
    """
    if hide_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    ax.tick_params(axis='both', labelsize=7, direction='out', width=0.5)


def add_grid(ax: plt.Axes, style: str = 'dashed') -> None:
    """
    Add subtle grid to axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    style : str
        'dashed' for dashed lines, 'solid' for solid lines
    """
    linestyle = '--' if style == 'dashed' else '-'
    ax.grid(True, linestyle=linestyle, linewidth=0.4, color='#DDDDDD', alpha=1.0, zorder=0)
    ax.set_axisbelow(True)


# ==============================================================================
# UPTAKE RANGE PLOT
# ==============================================================================

def plot_uptake_ranges(uptake_ranges: np.ndarray,
                       output_path: str = None,
                       output_format: str = None,
                       figsize: Tuple[float, float] = None,
                       show_plot: bool = False,
                       verbose: bool = True,
                       config: Dict[str, Any] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create uptake range scatter plot.

    Plots ranked deuterium uptake ranges as a scatter plot.

    Parameters
    ----------
    uptake_ranges : np.ndarray
        Array of uptake range values (will be sorted and ranked)
    output_path : str, optional
        Path to save the figure (extension will be adjusted based on output_format)
    output_format : str, optional
        Output format: 'tiff' (default), 'png', or 'pdf'
    figsize : tuple, optional
        Figure size in inches. If None, uses config or defaults.
    show_plot : bool
        Whether to display the plot interactively
    verbose : bool
        Print save path
    config : dict, optional
        Configuration dictionary for figure settings

    Returns
    -------
    tuple
        (fig, ax)
    """
    configure_matplotlib_uptake_range()

    # Get settings from config or use defaults
    settings = get_figure_settings(config, 'uptake_range') if config else {
        'figsize': UPTAKE_RANGE_FIGSIZE,
        'axis_title_fontsize': DEFAULT_UPTAKE_RANGE_AXIS_TITLE_FONTSIZE,
        'axis_number_fontsize': DEFAULT_UPTAKE_RANGE_AXIS_NUMBER_FONTSIZE,
        'dot_size': DEFAULT_UPTAKE_RANGE_DOT_SIZE,
        'exact_figsize': True,
        'dpi': DEFAULT_DPI
    }

    if figsize is None:
        figsize = settings['figsize']

    axis_title_fontsize = settings['axis_title_fontsize']
    axis_number_fontsize = settings['axis_number_fontsize']
    dot_size = settings['dot_size']
    exact_figsize = settings.get('exact_figsize', True)
    dpi = settings.get('dpi', DEFAULT_DPI)

    # Sort values and create ranks
    sorted_values = np.sort(uptake_ranges)
    ranks = np.arange(1, len(sorted_values) + 1)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Grid first (below data)
    add_grid(ax, style='solid')

    # Scatter plot
    ax.scatter(ranks, sorted_values, s=dot_size, color='royalblue', zorder=3)

    # Labels
    ax.set_xlabel('Rank', fontsize=axis_title_fontsize)
    ax.set_ylabel('Deuterium uptake ranges (Da)', fontsize=axis_title_fontsize)

    # Axis limits
    ax.set_xlim(0, len(ranks) + 25)
    ax.set_ylim(0, sorted_values.max() * 1.05)

    # Axis number styling
    ax.tick_params(axis='both', which='both', labelsize=axis_number_fontsize, width=0.5)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        actual_path = save_figure(fig, output_path, output_format=output_format,
                                  dpi=dpi, figure_type='uptake_range', exact_size=exact_figsize)
        if verbose:
            print(f"  Uptake range figure saved: {actual_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_uptake_ranges_from_df(df: pd.DataFrame,
                               column_name: str = 'Uptake Range (Da)',
                               output_path: str = None,
                               output_format: str = None,
                               figsize: Tuple[float, float] = None,
                               show_plot: bool = False,
                               verbose: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create uptake range scatter plot from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing uptake range data
    column_name : str
        Name of the column containing uptake range values
    output_path : str, optional
        Path to save the figure
    output_format : str, optional
        Output format: 'tiff' (default), 'png', or 'pdf'
    figsize : tuple, optional
        Figure size in inches
    show_plot : bool
        Whether to display the plot interactively
    verbose : bool
        Print save path

    Returns
    -------
    tuple
        (fig, ax)
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found. Available: {list(df.columns)}")

    uptake_ranges = df[column_name].dropna().astype(float).values

    return plot_uptake_ranges(
        uptake_ranges=uptake_ranges,
        output_path=output_path,
        output_format=output_format,
        figsize=figsize,
        show_plot=show_plot,
        verbose=verbose
    )


# ==============================================================================
# COMPLETE ENUMERATION PLOT
# ==============================================================================

def plot_complete_enumeration_figure(precomputed_df: pd.DataFrame,
                                      config: Dict[str, Any],
                                      output_dir: str,
                                      output_format: str = None,
                                      save_fig: bool = True,
                                      show_plot: bool = False,
                                      figsize: Tuple[float, float] = None,
                                      panel_label: str = None,
                                      verbose: bool = True) -> Dict[str, Any]:
    """
    Generate figure for complete enumeration data.

    Parameters
    ----------
    precomputed_df : pd.DataFrame
        Precomputed statistics DataFrame with columns:
        delta_d, ci_lp, ci_up, ci_lp_up_abs_max
    config : dict
        Configuration dictionary for figure settings
    output_dir : str
        Directory to save the figure
    output_format : str, optional
        Output format: 'tiff' (default), 'png', or 'pdf'
    save_fig : bool
        Whether to save the figure
    show_plot : bool
        Whether to display the plot interactively
    figsize : tuple, optional
        Figure size in inches. If None, uses config settings.
    panel_label : str, optional
        Panel label (e.g., "A", "B") for multi-panel figures
    verbose : bool
        Print status messages

    Returns
    -------
    dict
        Results dictionary with:
        - eac: Equivalence Acceptance Criterion
        - delta_d_limit: Maximum |delta_D|
        - n_points: Number of data points
        - figure_path: Path to saved figure (or None)
    """
    configure_matplotlib_publication()

    # Calculate limits from precomputed data
    eac = float(precomputed_df['ci_lp_up_abs_max'].max())
    delta_d_limit = float(np.abs(precomputed_df['delta_d']).max())
    n_points = len(precomputed_df)

    # Get settings from config (uses 'resampling' settings for consistent style)
    settings = get_figure_settings(config, 'resampling') if config else {
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

    if figsize is None:
        figsize = settings['figsize']

    # Extract and sort data by delta_d for ranking
    plot_data = precomputed_df[['delta_d', 'ci_lp', 'ci_up', 'ci_lp_up_abs_max']].copy()
    sorted_data = plot_data.sort_values('delta_d').reset_index(drop=True)

    delta_d = sorted_data['delta_d'].values
    ci_lp = sorted_data['ci_lp'].values
    ci_up = sorted_data['ci_up'].values
    rank = np.arange(n_points)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Fast plotting: Use LineCollection for CI bars (gray vertical lines)
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
    annotation_text = (
        f'EAC = {format_report_limit(eac, config)}    '
        f'$\\Delta D_{{\\mathrm{{limit}}}}$ = {format_report_limit(delta_d_limit, config)}'
    )
    ax.text(1.0, 1.02, annotation_text,
            transform=ax.transAxes,
            fontsize=result_text_fontsize,
            verticalalignment='bottom',
            horizontalalignment='right')

    plt.tight_layout()

    # Generate output path
    fig_path = None
    if save_fig and output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = get_format_extension(output_format)
        fig_filename = f"complete_enumeration_{n_points}_{timestamp}{ext}"
        fig_path = os.path.join(output_dir, fig_filename)

        actual_path = save_figure(fig, fig_path, output_format=output_format,
                                  dpi=dpi, figure_type='resampling', exact_size=exact_figsize)
        fig_path = actual_path

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        'eac': eac,
        'delta_d_limit': delta_d_limit,
        'n_points': n_points,
        'figure_path': fig_path
    }
