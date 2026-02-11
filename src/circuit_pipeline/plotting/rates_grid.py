from __future__ import annotations

from pathlib import Path
import numpy as np
from circuit_pipeline.plotting._backend import *  # noqa: F401

def save_rates_grid_for_alphas(
    *,
    time: np.ndarray,
    rates_by_alpha: list[dict[str, np.ndarray]],
    alfa_vec: np.ndarray,
    output_path: Path,
    dpi: int = 150,
    close: bool = True,
) -> None:
    """
    Save a grid of rate traces for multiple alphas.

    rates_by_alpha: list of dicts with keys: "rE", "rP", "rS", "rI"
                    each value is a 1D array same length as time.
    alfa_vec: array of alphas (same length as rates_by_alpha)
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    alfa_vec = np.asarray(alfa_vec, dtype=float)
    time = np.asarray(time, dtype=float)

    n = len(alfa_vec)
    if n == 0:
        raise ValueError("alfa_vec must contain at least one alpha.")

    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 3), squeeze=False)
    flat = axes.flatten()

    for i, alfa in enumerate(alfa_vec):
        ax = flat[i]
        r = rates_by_alpha[i]

        ax.plot(time, r["rE"], label=r"$r_E$")
        ax.plot(time, r["rP"], label=r"$r_P$")
        ax.plot(time, r["rS"], label=r"$r_S$")
        ax.plot(time, r["rI"], label=r"$r_I$")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$r_X$ (Hz)")
        ax.legend().set_title(rf"$\alpha$ = {alfa:g}")
        ax.set_xlim([float(time[0]), float(time[-1])])

    # turn off unused panels
    for j in range(n, len(flat)):
        flat[j].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if close:
        plt.close(fig)
