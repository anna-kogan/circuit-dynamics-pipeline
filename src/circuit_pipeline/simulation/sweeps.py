from __future__ import annotations

import numpy as np

from .dynamics import simulate_dynamics


def sweep_rP0_delta_rE(
    *,
    # ---- required base model params for simulate_dynamics ----
    W: np.ndarray,
    interneuron_name: str,
    wES_0: float,
    tau: np.ndarray,
    rS0: float,
    rI0: float,
    I_mod_I: float,
    power: float,
    mult_f: float,
    # ---- sweep params ----
    mod_onset_t: float = 50.0,
    rE0: float,
    alfa: float,
    rP0_vec: np.ndarray,
    dt: float,
    end_sim: float,
    progress: bool = False,
) -> np.ndarray:
    """
    Returns delta_rE for each rP0 in rP0_vec.
    Output shape: (len(rP0_vec),)
    """
    rP0_vec = np.asarray(rP0_vec, dtype=float)
    delta_rE = np.zeros_like(rP0_vec, dtype=float)

    n = len(rP0_vec)
    for i, rP0 in enumerate(rP0_vec):
        delta = simulate_dynamics(
            W=W,
            interneuron_name=interneuron_name,
            wES_0=float(wES_0),
            tau=tau,
            rS0=float(rS0),
            rI0=float(rI0),
            I_mod_I=float(I_mod_I),
            power=float(power),
            mult_f=float(mult_f),
            rE0=float(rE0),
            rP0=float(rP0),
            alfa=float(alfa),
            dt=float(dt),
            end_sim=float(end_sim),
            mod_onset_t=float(mod_onset_t),
            only_deltas=True,
        )
        delta_rE[i] = float(delta[0])

        if progress:
            perc = 100 * i / max(n, 1)
            if perc % 5 == 0:
                print(f"Finished {np.floor(perc)}%", end="\r")

    return delta_rE


def sweep_alpha_delta_rates(
    *,
    # ---- required base model params for simulate_dynamics ----
    W: np.ndarray,
    interneuron_name: str,
    wES_0: float,
    tau: np.ndarray,
    rS0: float,
    rI0: float,
    I_mod_I: float,
    power: float,
    mult_f: float,
    # ---- sweep params ----
    mod_onset_t: float = 50.0,
    rE0: float,
    rP0: float,
    alfa_vec: np.ndarray,
    dt: float,
    end_sim: float,
    progress: bool = False,
) -> np.ndarray:
    """
    Returns delta rates for each alpha.
    Output shape: (4, len(alfa_vec)) for [E,P,S,I].
    """
    alfa_vec = np.asarray(alfa_vec, dtype=float)
    out = np.zeros((4, len(alfa_vec)), dtype=float)

    n = len(alfa_vec)
    for i, alfa in enumerate(alfa_vec):
        out[:, i] = np.asarray(
            simulate_dynamics(
                W=W,
                interneuron_name=interneuron_name,
                wES_0=float(wES_0),
                tau=tau,
                rS0=float(rS0),
                rI0=float(rI0),
                I_mod_I=float(I_mod_I),
                power=float(power),
                mult_f=float(mult_f),
                rE0=float(rE0),
                rP0=float(rP0),
                alfa=float(alfa),
                dt=float(dt),
                end_sim=float(end_sim),
                mod_onset_t=float(mod_onset_t),
                only_deltas=True,
            ),
            dtype=float,
        )

        if progress:
            perc = 100 * i / max(n, 1)
            if perc % 10 == 0:
                print(f"Finished {np.floor(perc)}%", end="\r")

    return out
