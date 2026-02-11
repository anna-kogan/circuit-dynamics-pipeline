from __future__ import annotations

from pathlib import Path
import numpy as np

from circuit_pipeline.model.linear_response import CircuitParams, compute_point
from circuit_pipeline.pipeline.artifacts import save_npz, Timer


def run_linear_grid(
    *,
    params: CircuitParams,
    max_rEP: float,
    step_rX: float,
    alfa: float,
    save_npz_path: Path | None = None,
    force: bool = False,
    progress: bool = False,
) -> "LinearGridResult":
    """
    Build rE/rP grid, compute compute_point at each cell, return arrays.
    Optionally save to NPZ.

    Resume behavior:
      - if save_npz_path exists and force=False -> loads + returns saved results
    """
    if save_npz_path is not None:
        save_npz_path = Path(save_npz_path)
        if save_npz_path.exists() and not force:
            data = np.load(save_npz_path, allow_pickle=False)
            return LinearGridResult.from_npz(data)

    max_rEP = float(max_rEP)
    step_rX = float(step_rX)
    alfa = float(alfa)

    rE_vec = np.round(np.arange(max_rEP, 0.5 - step_rX, -step_rX), 2)
    rP_vec = np.round(np.arange(0.5, max_rEP + step_rX, step_rX), 2)

    shape = (len(rE_vec), len(rP_vec))

    f_gain_num = np.zeros(shape, dtype=float)
    f_gain_num_mod = np.zeros(shape, dtype=float)
    f_maxEVs_num = np.zeros(shape, dtype=float)
    f_maxEVs_num_mod = np.zeros(shape, dtype=float)
    f_maxImEVs_num = np.zeros(shape, dtype=float)
    f_oscMetric_num = np.zeros(shape, dtype=float)
    f_oscMetric_num_mod = np.zeros(shape, dtype=float)
    f_modI_rE_num = np.zeros(shape, dtype=float)
    f_modI_rP_num = np.zeros(shape, dtype=float)
    f_modI_rS_num = np.zeros(shape, dtype=float)
    f_modI_rI_num = np.zeros(shape, dtype=float)

    total = len(rE_vec) * len(rP_vec)
    k = 0

    with Timer() as t:
        for i, rE in enumerate(rE_vec):
            for j, rP in enumerate(rP_vec):
                res = compute_point(params, float(rE), float(rP), alfa)

                f_gain_num[i, j] = res.gain
                f_gain_num_mod[i, j] = res.gain_mod
                f_maxEVs_num[i, j] = res.max_ev
                f_maxEVs_num_mod[i, j] = res.max_ev_mod
                f_maxImEVs_num[i, j] = res.max_im_ev
                f_oscMetric_num[i, j] = res.osc_metric
                f_oscMetric_num_mod[i, j] = res.osc_metric_mod
                f_modI_rE_num[i, j] = res.modI_rE
                f_modI_rP_num[i, j] = res.modI_rP
                f_modI_rS_num[i, j] = res.modI_rS
                f_modI_rI_num[i, j] = res.modI_rI

                if progress:
                    k += 1
                    perc = 100 * k / max(total, 1)
                    if perc % 5 == 0:
                        print(f"Finished {np.floor(perc)}%", end="\r")

    out = LinearGridResult(
        rE_vec=rE_vec,
        rP_vec=rP_vec,
        f_gain_num=f_gain_num,
        f_gain_num_mod=f_gain_num_mod,
        f_maxEVs_num=f_maxEVs_num,
        f_maxEVs_num_mod=f_maxEVs_num_mod,
        f_maxImEVs_num=f_maxImEVs_num,
        f_oscMetric_num=f_oscMetric_num,
        f_oscMetric_num_mod=f_oscMetric_num_mod,
        f_modI_rE_num=f_modI_rE_num,
        f_modI_rP_num=f_modI_rP_num,
        f_modI_rS_num=f_modI_rS_num,
        f_modI_rI_num=f_modI_rI_num,
        elapsed_s=t.elapsed_s,
    )

    if save_npz_path is not None:
        save_npz(save_npz_path, **out.to_npz_dict())

    return out


# ------------------------------------------------------

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearGridResult:
    rE_vec: np.ndarray
    rP_vec: np.ndarray
    f_gain_num: np.ndarray
    f_gain_num_mod: np.ndarray
    f_maxEVs_num: np.ndarray
    f_maxEVs_num_mod: np.ndarray
    f_maxImEVs_num: np.ndarray
    f_oscMetric_num: np.ndarray
    f_oscMetric_num_mod: np.ndarray
    f_modI_rE_num: np.ndarray
    f_modI_rP_num: np.ndarray
    f_modI_rS_num: np.ndarray
    f_modI_rI_num: np.ndarray
    elapsed_s: float | None = None

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        # elapsed_s stored separately in manifest
        return {
            "rE_vec": self.rE_vec,
            "rP_vec": self.rP_vec,
            "f_gain_num": self.f_gain_num,
            "f_gain_num_mod": self.f_gain_num_mod,
            "f_maxEVs_num": self.f_maxEVs_num,
            "f_maxEVs_num_mod": self.f_maxEVs_num_mod,
            "f_maxImEVs_num": self.f_maxImEVs_num,
            "f_oscMetric_num": self.f_oscMetric_num,
            "f_oscMetric_num_mod": self.f_oscMetric_num_mod,
            "f_modI_rE_num": self.f_modI_rE_num,
            "f_modI_rP_num": self.f_modI_rP_num,
            "f_modI_rS_num": self.f_modI_rS_num,
            "f_modI_rI_num": self.f_modI_rI_num,
        }

    @staticmethod
    def from_npz(data: np.lib.npyio.NpzFile) -> "LinearGridResult":
        return LinearGridResult(
            rE_vec=np.asarray(data["rE_vec"], dtype=float),
            rP_vec=np.asarray(data["rP_vec"], dtype=float),
            f_gain_num=np.asarray(data["f_gain_num"], dtype=float),
            f_gain_num_mod=np.asarray(data["f_gain_num_mod"], dtype=float),
            f_maxEVs_num=np.asarray(data["f_maxEVs_num"], dtype=float),
            f_maxEVs_num_mod=np.asarray(data["f_maxEVs_num_mod"], dtype=float),
            f_maxImEVs_num=np.asarray(data["f_maxImEVs_num"], dtype=float),
            f_oscMetric_num=np.asarray(data["f_oscMetric_num"], dtype=float),
            f_oscMetric_num_mod=np.asarray(data["f_oscMetric_num_mod"], dtype=float),
            f_modI_rE_num=np.asarray(data["f_modI_rE_num"], dtype=float),
            f_modI_rP_num=np.asarray(data["f_modI_rP_num"], dtype=float),
            f_modI_rS_num=np.asarray(data["f_modI_rS_num"], dtype=float),
            f_modI_rI_num=np.asarray(data["f_modI_rI_num"], dtype=float),
            elapsed_s=None,
        )
