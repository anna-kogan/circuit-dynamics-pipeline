import numpy as np
from pathlib import Path

from circuit_pipeline.model.linear_response import CircuitParams
from circuit_pipeline.pipeline.run_grid import run_linear_grid


def test_run_linear_grid_resume(tmp_path: Path):
    params = CircuitParams(
        interneuron_name="vip",
        W=np.eye(4),
        tau=np.ones(4),
        rS0=2.0,
        rI0=2.0,
        step_rX=0.5,
        I_stim_E=0.3,
        I_stim_P=0.3,
        I_mod_I=0.3,
        power=2.0,
        mult_f=0.25,
    )

    out = tmp_path / "grid.npz"

    g1 = run_linear_grid(params=params, max_rEP=2.0, step_rX=0.5, alfa=2.0, save_npz_path=out, force=True)
    assert out.exists()

    # second run load, not recompute
    g2 = run_linear_grid(params=params, max_rEP=2.0, step_rX=0.5, alfa=2.0, save_npz_path=out, force=False)
    assert np.allclose(g1.f_gain_num, g2.f_gain_num, equal_nan=True)
    

def test_run_linear_grid_vectors_resume(tmp_path):
    params = CircuitParams(
        interneuron_name="vip",
        W=np.eye(4),
        tau=np.ones(4),
        rS0=2.0,
        rI0=2.0,
        step_rX=0.5,
        I_stim_E=0.3,
        I_stim_P=0.3,
        I_mod_I=0.3,
        power=2.0,
        mult_f=0.25,
    )

    out = tmp_path / "grid.npz"

    g1 = run_linear_grid(params=params, max_rEP=2.0, step_rX=0.5, alfa=2.0, save_npz_path=out, force=True)
    g2 = run_linear_grid(params=params, max_rEP=2.0, step_rX=0.5, alfa=2.0, save_npz_path=out, force=False)

    np.testing.assert_allclose(g1.rE_vec, g2.rE_vec)
    np.testing.assert_allclose(g1.rP_vec, g2.rP_vec)

