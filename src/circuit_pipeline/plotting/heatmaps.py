from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import numpy as np
from circuit_pipeline.plotting._backend import *  # noqa: F401

from .styles import make_cmap


@dataclass(frozen=True)
class GridPlotData:
    rE_vec: np.ndarray
    rP_vec: np.ndarray
    gain: np.ndarray
    max_ev: np.ndarray | None = None
    osc_metric: np.ndarray | None = None
    gain_mod: np.ndarray | None = None
    max_ev_mod: np.ndarray | None = None
    osc_metric_mod: np.ndarray | None = None


def plot_grid_heatmaps(
    data,
    *,
    output_path: Path,
    title: str | None = None,
    dpi: int = 150,
    close: bool = True,
):
    """
    Plot grid heatmaps and save to disk.
    """
    import matplotlib.pyplot as plt

    cmap = make_cmap()

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        data.gain,
        aspect="auto",
        origin="upper",
        cmap=cmap,
    )

    ax.set_title(title or "Gain heatmap")
    ax.set_xlabel("rP")
    ax.set_ylabel("rE")

    fig.colorbar(im, ax=ax, label="gain")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if close:
        plt.close(fig)
