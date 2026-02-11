from __future__ import annotations

from pathlib import Path
import numpy as np
import yaml

from circuit_pipeline.legacy.extended_circuit import ExtendedCircuit


def build_circuit_from_yaml(cfg: dict) -> ExtendedCircuit:
    c = cfg["circuit"]
    return ExtendedCircuit(
        interneuron_name=c["interneuron_name"],
        W=np.array(c["W"], dtype=float),
        tau=np.array(c["tau"], dtype=float),
        rS0=float(c["rS0"]),
        rI0=float(c["rI0"]),
        step_rX=float(c["step_rX"]),
        I_stim_E=float(c["I_stim_E"]),
        I_stim_P=float(c["I_stim_P"]),
        I_mod_I=float(c["I_mod_I"]),
        power=float(c["power"]),
        mult_f=float(c["mult_f"]),
    )


def test_reference_grid_matches_npz():
    root = Path(__file__).resolve().parents[1]

    cfg_path = root / "configs" / "reference_dynamics.yaml"
    npz_path = root / "reference" / "reference_dynamics_old.npz"

    assert cfg_path.exists(), f"Missing config: {cfg_path}"
    assert npz_path.exists(), f"Missing reference npz: {npz_path}"

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    grid = cfg["grid"]

    c = build_circuit_from_yaml(cfg)

    outputs = c.calculate_linear(
        max_rEP=float(grid["max_rEP"]),
        alfa=float(grid["alfa"]),
    )

    (
        f_gain_num,
        f_gain_num_mod,
        f_maxEVs_num,
        f_maxEVs_num_mod,
        f_maxImEVs_num,
        f_oscMetric_num,
        f_oscMetric_num_mod,
        f_modI_rE_num,
        f_modI_rP_num,
        f_modI_rS_num,
        f_modI_rI_num,
        rE_vec,
        rP_vec,
    ) = outputs

    ref = np.load(npz_path)

    # sanity: grid vectors should match too
    assert "rE_vec" in ref and "rP_vec" in ref, "Reference NPZ missing rE_vec/rP_vec"
    np.testing.assert_allclose(np.asarray(rE_vec, dtype=float), ref["rE_vec"], rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(np.asarray(rP_vec, dtype=float), ref["rP_vec"], rtol=0, atol=0, equal_nan=True)

    # Main regression checks
    np.testing.assert_allclose(f_gain_num, ref["f_gain_num"], rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(f_gain_num_mod, ref["f_gain_num_mod"], rtol=1e-12, atol=1e-12, equal_nan=True)

    np.testing.assert_allclose(f_maxEVs_num, ref["f_maxEVs_num"], rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(f_maxEVs_num_mod, ref["f_maxEVs_num_mod"], rtol=1e-12, atol=1e-12, equal_nan=True)

    np.testing.assert_allclose(f_maxImEVs_num, ref["f_maxImEVs_num"], rtol=1e-12, atol=1e-12, equal_nan=True)

    np.testing.assert_allclose(f_oscMetric_num, ref["f_oscMetric_num"], rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(f_oscMetric_num_mod, ref["f_oscMetric_num_mod"], rtol=1e-12, atol=1e-12, equal_nan=True)

    np.testing.assert_allclose(f_modI_rE_num, ref["f_modI_rE_num"], rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(f_modI_rP_num, ref["f_modI_rP_num"], rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(f_modI_rS_num, ref["f_modI_rS_num"], rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(f_modI_rI_num, ref["f_modI_rI_num"], rtol=1e-12, atol=1e-12, equal_nan=True)
