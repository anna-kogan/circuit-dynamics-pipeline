from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from circuit_pipeline.model.linear_response import CircuitParams
from circuit_pipeline.pipeline.artifacts import make_run_dir, save_manifest
from circuit_pipeline.pipeline.run_grid import run_linear_grid
from circuit_pipeline.plotting.heatmaps import GridPlotData, plot_grid_heatmaps


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_circuit_params(cfg: dict) -> tuple[CircuitParams, float, float, float]:
    circuit = cfg["circuit"]
    grid = cfg["grid"]

    W = np.array(circuit["W"], dtype=float)
    tau = np.array(circuit["tau"], dtype=float)

    params = CircuitParams(
        interneuron_name=str(circuit["interneuron_name"]),
        W=W,
        tau=tau,
        rS0=float(circuit["rS0"]),
        rI0=float(circuit["rI0"]),
        I_stim_E=float(circuit.get("I_stim_E", 0.0)),
        I_stim_P=float(circuit.get("I_stim_P", 0.0)),
        I_mod_I=float(circuit.get("I_mod_I", 0.0)),
        power=float(circuit.get("power", 2.0)),
        mult_f=float(circuit.get("mult_f", 0.25)),
    )

    max_rEP = float(grid["max_rEP"])
    alfa = float(grid["alfa"])
    step_rX = float(circuit["step_rX"])

    return params, max_rEP, alfa, step_rX


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="circuit-grid")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--run-root", default="runs", help="Root folder for output runs/")
    p.add_argument("--run-name", default=None, help="Optional run folder name (default: timestamp)")
    p.add_argument("--force", action="store_true", help="Recompute even if NPZ exists")
    p.add_argument("--progress", action="store_true", help="Print progress while running")
    return p


def main() -> int:
    args = build_parser().parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)

    params, max_rEP, alfa, step_rX = parse_circuit_params(cfg)

    run_dir = make_run_dir(root=args.run_root, run_name=args.run_name)

    grid_npz = run_dir / "grid.npz"
    heatmap_path = run_dir / "heatmap_gain.png"
    manifest_path = run_dir / "manifest.json"

    result = run_linear_grid(
        params=params,
        max_rEP=max_rEP,
        step_rX=step_rX,
        alfa=alfa,
        save_npz_path=grid_npz,
        force=args.force,
        progress=args.progress,
    )

    plot_grid_heatmaps(
        GridPlotData(
            rE_vec=result.rE_vec,
            rP_vec=result.rP_vec,
            gain=result.f_gain_num,
            max_ev=result.f_maxEVs_num,
            osc_metric=result.f_oscMetric_num,
        ),
        output_path=heatmap_path,
        title=f"Gain heatmap (alfa={alfa})",
        dpi=150,
        close=True,
    )

    save_manifest(
        path=manifest_path,
        config={"config_path": str(config_path), "config": cfg},
        outputs={"grid_npz": grid_npz, "heatmap_gain": heatmap_path},
        timings={"grid_elapsed_s": result.elapsed_s},
        extra={"command": "circuit-grid"},
    )

    print(f"Run saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())