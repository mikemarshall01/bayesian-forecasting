"""House chart style, shared across every notebook so the whole repo (and the wider
handbook) looks consistent. Import and call ``use_house_style()`` once at the top of a
notebook, then use the ``PALETTE`` colours for series.

The goal is calm, readable, presentation-grade charts: light background, soft grid,
no chart-junk, one accent colour per idea.
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt

# One primary, one accent, plus muted supporting tones. Re-used everywhere.
PALETTE = {
    "ink":    "#1b1f24",  # near-black text / primary line
    "blue":   "#2f6fed",  # primary accent
    "teal":   "#159a8c",  # secondary
    "amber":  "#e2a32b",  # signal / highlight
    "red":    "#d1495b",  # stress / short
    "green":  "#2e8b57",  # long / health
    "grey":   "#9aa3ad",  # muted / reference lines
    "grid":   "#e7e9ee",
}
CYCLE = [PALETTE[c] for c in ("blue", "amber", "teal", "red", "green", "grey")]


def use_house_style() -> None:
    """Apply the repo-wide matplotlib style. Call once per notebook."""
    mpl.rcParams.update({
        "figure.figsize":     (10, 4.5),
        "figure.dpi":         110,
        "savefig.dpi":        140,
        "savefig.bbox":       "tight",
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.labelsize":     11,
        "axes.edgecolor":     PALETTE["grey"],
        "axes.linewidth":     0.8,
        "axes.grid":          True,
        "axes.axisbelow":     True,
        "axes.prop_cycle":    mpl.cycler(color=CYCLE),
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.color":         PALETTE["grid"],
        "grid.linewidth":     0.9,
        "legend.frameon":     False,
        "legend.fontsize":    10,
        "xtick.color":        PALETTE["ink"],
        "ytick.color":        PALETTE["ink"],
        "text.color":         PALETTE["ink"],
        "axes.labelcolor":    PALETTE["ink"],
        "axes.titlecolor":    PALETTE["ink"],
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
    })


def titled(ax, title: str, subtitle: str | None = None):
    """Bold left title with an optional muted subtitle on its own line beneath it.

    The subtitle is offset in *points* (not axes fractions) and the title is given
    matching head-room, so the two never overlap regardless of figure size.
    """
    if subtitle:
        ax.set_title(title, loc="left", pad=24)
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 7), textcoords="offset points",
                    fontsize=9.5, color=PALETTE["grey"], ha="left", va="bottom")
    else:
        ax.set_title(title, loc="left", pad=10)
    return ax
