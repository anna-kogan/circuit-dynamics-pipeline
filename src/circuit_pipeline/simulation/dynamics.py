from __future__ import annotations

import numpy as np


def simulate_dynamics(
    *,
    W: np.ndarray,
    interneuron_name: str,
    wES_0: float,
    tau: np.ndarray,          # shape (4,) or (4,1)
    rS0: float,
    rI0: float,
    I_mod_I: float,
    power: float,
    mult_f: float,
    rE0: float,
    rP0: float,
    alfa: float,
    dt: float,
    end_sim: float,
    mod_onset_t: float = 50.0,
    only_deltas: bool = False,
) -> tuple:
    """
    Calculates the real dynamics of the system with fixed SST and NDNF/VIP rates and passed E and PV rates.
    Also can calculate only changes of steady state rates before and after NDNF/VIP modulation.

    Returns:
      - if only_deltas=False:
        (rE_save, rP_save, rS_save, rI_save, wES_save, wIS_save, delta_r_vec)
      - if only_deltas=True:
        delta_r_vec
    """
    # Ensure shapes match original behavior
    tau_col = np.asarray(tau, dtype=float).reshape(4, 1)

    # to account for inhibitory weights sign
    W_sim = np.array(W, dtype=float, copy=True) * np.array([1, -1, -1, -1], dtype=float)

    # Initial wES depends on baseline rI0
    if interneuron_name == 'ndnf':
        W_sim[0, 2] = -float(wES_0) * np.exp(-float(rI0) / float(alfa))

    r = np.array([[rE0], [rP0], [rS0], [rI0]], dtype=float)

    I_mod = np.array([[0.0], [0.0], [0.0], [float(I_mod_I)]], dtype=float)

    I0 = (r / float(mult_f)) ** (1.0 / float(power)) - (W_sim @ r)
    I = I0

    # Set initial values of the rates and weights
    if not only_deltas:
        rE_save = [float(r[0, 0])]
        rP_save = [float(r[1, 0])]
        rS_save = [float(r[2, 0])]
        rI_save = [float(r[3, 0])]
        wES_save = [float(-W_sim[0, 2])]
        wIS_save = [float(-W_sim[3, 2])]

    for t in np.arange(dt, end_sim + dt, dt):
        if np.isclose(t, mod_onset_t, atol=dt):
            I += I_mod

        q = (W_sim @ r) + I
        r += dt * ((float(mult_f) * (q ** float(power))) - r) / tau_col

        # Update wES based on current rI
        if interneuron_name == 'ndnf':
            W_sim[0, 2] = -float(wES_0) * np.exp(-float(r[3, 0]) / float(alfa))

        if not only_deltas:
            rE_save.append(float(r[0, 0]))
            rP_save.append(float(r[1, 0]))
            rS_save.append(float(r[2, 0]))
            rI_save.append(float(r[3, 0]))
            wES_save.append(float(-W_sim[0, 2]))
            wIS_save.append(float(-W_sim[3, 2]))

    delta_r = r - np.array([[rE0], [rP0], [rS0], [rI0]], dtype=float)
    delta_r_vec = delta_r.reshape(4)

    if only_deltas:
        return delta_r_vec

    return (rE_save, rP_save, rS_save, rI_save, wES_save, wIS_save, delta_r_vec)
