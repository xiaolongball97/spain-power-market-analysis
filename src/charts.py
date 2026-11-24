"""Consistent, minimal chart styling for the project."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
# Categorical slots in fixed order (blue, orange, aqua, yellow)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def series_colors(n: int) -> list[str]:
    if n > len(SERIES):
        raise ValueError("Too many series for one chart: fold into 'Other' or facet")
    return SERIES[:n]


def styled_axes(title: str, xlabel: str, ylabel: str, size=(9, 4.5)):
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, color=TEXT, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return ax, fig


def line_chart(ax, frame, label_fmt="{}"):
    """Each column is a series; direct-labelled at the line end plus a legend."""
    colors = series_colors(frame.shape[1])
    for col, color in zip(frame.columns, colors):
        ax.plot(frame.index, frame[col], color=color, linewidth=2, label=label_fmt.format(col))
        ax.annotate(label_fmt.format(col), (frame.index[-1], frame[col].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    color=TEXT_2, fontsize=9)
    ax.legend(frameon=False, ncol=frame.shape[1], loc="upper left", fontsize=9)
    ax.margins(x=0.06)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.15)  # headroom so the legend never covers data


def bar_chart(ax, labels, values, series_names=None, value_fmt="{:.1f}",
              ylim=None, ref_line=None, legend_loc="upper left"):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n_series = values.shape[1]
    colors = series_colors(n_series)
    x = np.arange(len(labels))
    width = 0.8 / n_series
    for i in range(n_series):
        offset = (i - (n_series - 1) / 2) * width
        bars = ax.bar(x + offset, values[:, i], width=width - 0.03, color=colors[i],
                      label=series_names[i] if series_names else None)
        for b, v in zip(bars, values[:, i]):
            ax.annotate(value_fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3 if v >= 0 else -3), textcoords="offset points",
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=8, color=TEXT_2)
    ax.set_xticks(x, labels)
    if ylim:
        ax.set_ylim(*ylim)
    if ref_line is not None:
        ax.axhline(ref_line, color=TEXT_2, linewidth=1, linestyle="--")
    if series_names:
        ax.legend(frameon=False, loc=legend_loc, ncol=n_series, fontsize=9)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
