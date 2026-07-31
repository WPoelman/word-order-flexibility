from __future__ import annotations

import matplotlib as mpl

SINGLE_COLUMN_PT = 219
DOUBLE_COLUMN_PT = 455
POINTS_PER_INCH = 72.27
CANVAS_SCALE = {1: 1.09, 2: 1.04}


def figure_size(columns: int = 1, *, height_ratio: float = 0.72) -> tuple[float, float]:
    """Return a paper-sized figure in inches for one or two columns."""
    if columns not in (1, 2):
        raise ValueError("columns must be 1 or 2")
    points = SINGLE_COLUMN_PT if columns == 1 else DOUBLE_COLUMN_PT
    # Tight export removes the canvas margins. Slightly enlarge the source
    # canvas so the cropped artifact retains its intended column width.
    width = points * CANVAS_SCALE[columns] / POINTS_PER_INCH
    return width, width * height_ratio


def paper_style() -> dict:
    """Matplotlib settings whose point sizes match the typeset paper."""
    return {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{newtxtext}",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "legend.framealpha": 1.0,
        "xtick.bottom": True,
        "ytick.left": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }


def format_decimal(value: float) -> str:
    """Format parameter values as .25, .5, .75, and 1."""
    if value == 0:
        return "0"
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text.removeprefix("0")


def apply_paper_style() -> None:
    mpl.rcParams.update(paper_style())
