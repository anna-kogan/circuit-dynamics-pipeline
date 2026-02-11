from pathlib import Path
import numpy as np

from circuit_pipeline.legacy.extended_circuit import ExtendedCircuit
from circuit_pipeline.pipeline.artifacts import make_run_dir, save_manifest


def main():
    # define the circuit
    W = np.array(
        [
            [0.8, 0.5, 0.3, 0.3],
            [1.0, 0.6, 0.8, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.0],
        ],
        dtype=float,
    )

    c = ExtendedCircuit(interneuron_name="ndnf", W=W, step_rX=0.5)

    # create a run dir
    run_dir = make_run_dir(root="runs", run_name=None)

    # compute grid
    grid_npz = run_dir / "grid.npz"

    (
        f_gain,
        f_gain_mod,
        f_maxEV,
        f_maxEV_mod,
        f_maxImEV,
        f_osc,
        f_osc_mod,
        f_modI_rE,
        f_modI_rP,
        f_modI_rS,
        f_modI_rI,
        rE_vec,
        rP_vec,
    ) = c.calculate_linear(
        max_rEP=2.0,
        alfa=10.0,
        save_npz_path=grid_npz,
        force=False,
        progress=True,
    )

    # plot artifacts
    heatmap_path = run_dir / "heatmap_gain.png"
    c.plot_heatmaps_with_arrows_and_scatter(
        rE_vec=rE_vec,
        rP_vec=rP_vec,
        gain=f_gain,
        max_ev=f_maxEV,
        osc_metric=f_osc,
        output_path=heatmap_path,
        title="Gain heatmap",
        dpi=150,
        close=True,
    )

    # manifest (what you ran + where outputs are)
    save_manifest(
        path=run_dir / "manifest.json",
        config={
            "interneuron_name": c.interneuron_name,
            "W": W,
            "tau": [c.tauE, c.tauP, c.tauS, c.tauI],
            "rS0": c.rS0,
            "rI0": c.rI0,
            "step_rX": c.step_rX,
            "I_mod_I": c.I_mod_I,
            "power": c.power,
            "mult_f": c.mult_f,
            "grid": {"max_rEP": 2.0, "alfa": 10.0},
        },
        outputs={
            "grid_npz": grid_npz,
            "heatmap_gain": heatmap_path,
        },
        timings={
        },
    )

    print(f"Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
