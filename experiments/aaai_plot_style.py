"""Shared plotting defaults for AAAI paper figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

AAAI_COLUMN_WIDTH = 3.3
AAAI_TEXT_WIDTH = 6.9


def apply_aaai_plot_style() -> None:
    """Use the minimum permitted AAAI text size and a paper-matched font."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
