import numpy as np
from src.circuit_pipeline.model.linear_compare import LinearCompareParams, compute_linear_delta_over_alpha


def test_linear_compare_shape():
    p = LinearCompareParams(
        interneuron_name="vip",

        wEE=1.0, wEP=0.1, wES_0=0.2, wEI=0.3,
        wPE=0.1, wPP=1.0, wPS=0.1, wPI=0.1,
        wSE=0.1, wSP=0.1, wSS=1.0, wSI=0.1,
        wIE=0.1, wIP=0.1, wIS=0.1, wII=1.0,

        rS0=2.0,
        rI0=2.0,

        power=2.0,
        mult_f=0.25,
        I_mod_I=0.3,
    )

    alfa_vec = np.array([1.0, 2.0, 3.0], dtype=float)
    out = compute_linear_delta_over_alpha(p, rE0=5.0, rP0=5.0, alfa_vec=alfa_vec, only_excitatory=True)
    assert out.shape == (4, len(alfa_vec))
