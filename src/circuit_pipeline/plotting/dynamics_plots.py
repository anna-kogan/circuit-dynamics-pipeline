from __future__ import annotations

from pathlib import Path
import numpy as np
from circuit_pipeline.plotting._backend import *  # noqa: F401


def plot_rate_weight_dynamics(
    *,
    time: np.ndarray,
    rE: np.ndarray,
    rP: np.ndarray,
    rS: np.ndarray,
    rI: np.ndarray,
    wES: np.ndarray,
    wIS: np.ndarray,
    alfa: float,
    end_sim: float,
    output_path: Path,
    dpi: int = 150,
    close: bool = True,
) -> None:
    """Saves a 2-panel dynamics plot: rates (top) and weights (bottom)."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(time, rE, "r", label="rE")
    ax[0].plot(time, rP, "b", label="rP")
    ax[0].plot(time, rS, "g", label="rS")
    ax[0].plot(time, rI, "m", label="rI")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("rX (Hz)")
    ax[0].legend().set_title(rf"$\alpha$ = {alfa}")
    ax[0].set_xlim([0, end_sim])

    ax[1].plot(time, wES, "r", label=r"$w_{ES}$")
    ax[1].plot(time, wIS, "m", label=r"$w_{IS}$")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Weight")
    ax[1].legend().set_title(rf"$\alpha$ = {alfa}")
    ax[1].set_xlim([0, end_sim])

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if close:
        plt.close(fig)


def plot_rates_for_alphas(
    *,
    time: np.ndarray,
    series_by_alpha: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    end_sim: float,
    output_path: Path,
    dpi: int = 150,
    close: bool = True,
) -> None:
    """
    Saves a grid of rate dynamics plots for multiple alphas.
    
    series_by_alpha: list of tuples (alfa, rE, rP, rS, rI)
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(series_by_alpha)
    nrows = int(np.ceil(n / 2))
    fig, ax = plt.subplots(nrows, 2, figsize=(12, nrows * 3))
    ax_flat = np.array(ax).reshape(-1)

    for i, (alfa, rE, rP, rS, rI) in enumerate(series_by_alpha):
        a = ax_flat[i]
        a.plot(time, rE, "r", label=r"$r_E$")
        a.plot(time, rP, "b", label=r"$r_P$")
        a.plot(time, rS, "g", label=r"$r_S$")
        a.plot(time, rI, "m", label=r"$r_I$")
        a.set_xlabel("Time (s)")
        a.set_ylabel(r"$r_X$ (Hz)")
        a.legend().set_title(rf"$\alpha$ = {alfa}")
        a.set_xlim([0, end_sim])

    # Hide unused panels
    for j in range(n, len(ax_flat)):
        ax_flat[j].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if close:
        plt.close(fig)
