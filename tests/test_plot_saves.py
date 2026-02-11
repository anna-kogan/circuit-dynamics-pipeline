from pathlib import Path
import numpy as np

from src.circuit_pipeline.legacy.extended_circuit import ExtendedCircuit


def test_plot_drE_rP0_fix_alpha_saves(tmp_path: Path):
    W = np.eye(4, dtype=float)
    c = ExtendedCircuit(interneuron_name="vip", W=W)

    out = tmp_path / "drE_vs_rP0.png"
    c.plot_drE_rP0_fix_alpha(
        alfa=2.0,
        rE0=5.0,
        output_path=out,
        title="test",
        rP0_vec=np.arange(0, 5, 1),
        dt=0.1,
        end_sim=0.2,
        progress=False,
        close=True,
    )

    assert out.exists()
    assert out.stat().st_size > 0
