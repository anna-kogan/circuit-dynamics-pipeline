import numpy as np
from circuit_pipeline.legacy.extended_circuit import ExtendedCircuit

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

(
    gain,
    gain_mod,
    max_ev,
    max_ev_mod,
    max_im_ev,
    osc,
    osc_mod,
    modE,
    modP,
    modS,
    modI,
    rE_vec,
    rP_vec,
) = c.calculate_linear(max_rEP=2.0, alfa=10.0)

output_path = "reference/test_gain.png"

c.plot_heatmaps_with_arrows_and_scatter(
    gain=gain,
    rE_vec=rE_vec,
    rP_vec=rP_vec,
    max_ev=max_ev,
    osc_metric=osc,
    output_path=output_path,
    title="Test gain",
)

print(f"Saved {output_path}")
