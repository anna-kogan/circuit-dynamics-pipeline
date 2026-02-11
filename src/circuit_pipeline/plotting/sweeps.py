from __future__ import annotations

from pathlib import Path
import numpy as np
from circuit_pipeline.plotting._backend import *  # noqa: F401


def plot_drE_vs_rP0(
    *,
    rP0_vec: np.ndarray,
    delta_rE: np.ndarray,
    title: str,
    output_path: Path,
    dpi: int = 150,
    close: bool = True,
) -> None:
    """Plots excitatory neuron population rate change against initial rate of PV value for fixed alpha and initial rate of E value"""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(rP0_vec, delta_rE, c="#610000", marker=".", s=10)
    ax.set_xlabel(r"$r_{P_0}\,[Hz]$")
    ax.set_ylabel(r"$\Delta r_E\,[Hz]$")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)


def plot_delta_rates_vs_alpha(
    *,
    alfa_vec: np.ndarray,
    delta_r: np.ndarray,
    title: str,
    output_path: Path,
    only_excitatory: bool = True,
    linear_compare: bool = False,
    linear_delta_r: np.ndarray | None = None,
    margins: float = 0.1,
    dpi: int = 150,
    close: bool = True,
) -> None:
    """
    delta_r: shape (4, n) where rows correspond to [E,P,S,I]
    linear_delta_r: optional, same shape as delta_r (used when linear_compare=True)
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(alfa_vec, delta_r[0, :], c="#750000", marker="v", s=10, label=r"$\Delta r_E$")

    if not only_excitatory:
        ax.scatter(alfa_vec, delta_r[1, :], c="#000066", marker="v", label=r"$\Delta r_P$")
        ax.scatter(alfa_vec, delta_r[2, :], c="#009900", marker="v", label=r"$\Delta r_S$")
        ax.scatter(alfa_vec, delta_r[3, :], c="#782747", marker="v", label=r"$\Delta r_I$")

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\Delta r_X\,[Hz]$")
    ax.set_title(title)

    if linear_compare and linear_delta_r is not None:
        axL = ax.twinx()
        axL.scatter(alfa_vec, linear_delta_r[0, :], c="#ff0000", marker=".", label=r"$\Delta r_E\,linear$")
        if not only_excitatory:
            axL.scatter(alfa_vec, linear_delta_r[1, :], c="#1a1aff", marker=".", label=r"$L_{PI}$")
            axL.scatter(alfa_vec, linear_delta_r[2, :], c="#00CD00", marker=".", label=r"$L_{SI}$")
            axL.scatter(alfa_vec, linear_delta_r[3, :], c="#d6457f", marker=".", label=r"$L_{II}$")
        axL.set_ylabel(r"$\Delta r_X$ for linearized system")

        (a1, a2), (l1, l2) = ax.get_ylim(), axL.get_ylim()
        axL.set_ylim(min(a1, l1) - margins, max(a2, l2) + margins)
        ax.set_ylim(min(a1, l1) - margins, max(a2, l2) + margins)

        lines2, labels2 = axL.get_legend_handles_labels()
        lines1, labels1 = ax.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="center left", bbox_to_anchor=(1.07, 0.5))
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1.07, 0.5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
