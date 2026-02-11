from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LinearCompareParams:
    interneuron_name: str

    # weights (scalars)
    wEE: float
    wEP: float
    wES_0: float
    wEI: float

    wPE: float
    wPP: float
    wPS: float
    wPI: float

    wSE: float
    wSP: float
    wSS: float
    wSI: float

    wIE: float
    wIP: float
    wIS: float
    wII: float

    # fixed baseline rates
    rS0: float
    rI0: float

    # IO params
    power: float
    mult_f: float

    # VIP/NDNF modulation amplitude
    I_mod_I: float


def compute_linear_delta_over_alpha(
    params: LinearCompareParams,
    *,
    rE0: float,
    rP0: float,
    alfa_vec: np.ndarray,
    only_excitatory: bool = True,
) -> np.ndarray:
    """
    Compute linearized delta-r over alpha sweep.

    Returns:
        linear_delta_r: np.ndarray shape (4, n_alpha), rows are [E,P,S,I].
        If only_excitatory=True, only row 0 is filled, others stay 0.
    """
    alfa_vec = np.asarray(alfa_vec, dtype=float)

    n = len(alfa_vec)
    L_save = np.zeros((4, n), dtype=float)
    linear_delta_r = np.zeros((4, n), dtype=float)

    p = params

    for i, alfa in enumerate(alfa_vec):
        rS = float(p.rS0)
        rI = float(p.rI0)
        rE = float(rE0)
        rP = float(rP0)

        # weight dependency correction
        if p.interneuron_name == "ndnf":
            wES = float(p.wES_0) * np.exp(-rI / float(alfa))
            pESN = (float(p.wES_0) / float(alfa)) * np.exp(-rI / float(alfa))
            wEI_correction = pESN * rS
            if wEI_correction > float(p.wEI):
                wEI_correction = float(p.wEI)
        elif p.interneuron_name == "vip":
            wES = float(p.wES_0)
            wEI_correction = 0.0
        else:
            raise ValueError('False name of the interneuron. Available options: "vip" or "ndnf".')

        # inputs (xE..xI)
        xE = (rE / p.mult_f) ** (1 / p.power) - (p.wEE * rE - p.wEP * rP - wES * rS - p.wEI * rI)
        xP = (rP / p.mult_f) ** (1 / p.power) - (p.wPE * rE - p.wPP * rP - p.wPS * rS - p.wPI * rI)
        xS = (rS / p.mult_f) ** (1 / p.power) - (p.wSE * rE - p.wSP * rP - p.wSS * rS - p.wSI * rI)
        xI = (rI / p.mult_f) ** (1 / p.power) - (p.wIE * rE - p.wIP * rP - p.wIS * rS - p.wII * rI)

        # gains (bE..bI)
        bE = p.power * p.mult_f * (p.wEE * rE - p.wEP * rP - wES * rS - p.wEI * rI + xE) ** (p.power - 1)
        bP = p.power * p.mult_f * (p.wPE * rE - p.wPP * rP - p.wPS * rS - p.wPI * rI + xP) ** (p.power - 1)
        bS = p.power * p.mult_f * (p.wSE * rE - p.wSP * rP - p.wSS * rS - p.wSI * rI + xS) ** (p.power - 1)
        bI = p.power * p.mult_f * (p.wIE * rE - p.wIP * rP - p.wIS * rS - p.wII * rI + xI) ** (p.power - 1)

        det_num = (
            -p.wEE*p.wII*p.wPP*p.wSS + p.wEE*p.wII*p.wPS*p.wSP
            + p.wEE*p.wIP*p.wPI*p.wSS - p.wEE*p.wIP*p.wPS*p.wSI
            - p.wEE*p.wIS*p.wPI*p.wSP + p.wEE*p.wIS*p.wPP*p.wSI
            + p.wEI*p.wIE*p.wPP*p.wSS - p.wEI*p.wIE*p.wPS*p.wSP
            - p.wEI*p.wIP*p.wPE*p.wSS + p.wEI*p.wIP*p.wPS*p.wSE
            + p.wEI*p.wIS*p.wPE*p.wSP - p.wEI*p.wIS*p.wPP*p.wSE
            - p.wEP*p.wIE*p.wPI*p.wSS + p.wEP*p.wIE*p.wPS*p.wSI
            + p.wEP*p.wII*p.wPE*p.wSS - p.wEP*p.wII*p.wPS*p.wSE
            - p.wEP*p.wIS*p.wPE*p.wSI + p.wEP*p.wIS*p.wPI*p.wSE
            + wES*p.wIE*p.wPI*p.wSP - wES*p.wIE*p.wPP*p.wSI
            - wES*p.wII*p.wPE*p.wSP + wES*p.wII*p.wPP*p.wSE
            + wES*p.wIP*p.wPE*p.wSI - wES*p.wIP*p.wPI*p.wSE
            - p.wIE*p.wPP*p.wSS*wEI_correction + p.wIE*p.wPS*p.wSP*wEI_correction
            + p.wIP*p.wPE*p.wSS*wEI_correction - p.wIP*p.wPS*p.wSE*wEI_correction
            - p.wIS*p.wPE*p.wSP*wEI_correction + p.wIS*p.wPP*p.wSE*wEI_correction
            - p.wEE*p.wII*p.wPP/bS + p.wEE*p.wIP*p.wPI/bS
            + p.wEI*p.wIE*p.wPP/bS - p.wEI*p.wIP*p.wPE/bS
            - p.wEP*p.wIE*p.wPI/bS + p.wEP*p.wII*p.wPE/bS
            - p.wIE*p.wPP*wEI_correction/bS + p.wIP*p.wPE*wEI_correction/bS
            - p.wEE*p.wII*p.wSS/bP + p.wEE*p.wIS*p.wSI/bP
            + p.wEI*p.wIE*p.wSS/bP - p.wEI*p.wIS*p.wSE/bP
            - wES*p.wIE*p.wSI/bP + wES*p.wII*p.wSE/bP
            - p.wIE*p.wSS*wEI_correction/bP + p.wIS*p.wSE*wEI_correction/bP
            - p.wEE*p.wII/(bP*bS) + p.wEI*p.wIE/(bP*bS) - p.wIE*wEI_correction/(bP*bS)
            - p.wEE*p.wPP*p.wSS/bI + p.wEE*p.wPS*p.wSP/bI
            + p.wEP*p.wPE*p.wSS/bI - p.wEP*p.wPS*p.wSE/bI
            - wES*p.wPE*p.wSP/bI + wES*p.wPP*p.wSE/bI
            - p.wEE*p.wPP/(bI*bS) + p.wEP*p.wPE/(bI*bS)
            - p.wEE*p.wSS/(bI*bP) + wES*p.wSE/(bI*bP)
            - p.wEE/(bI*bP*bS)
            + p.wII*p.wPP*p.wSS/bE - p.wII*p.wPS*p.wSP/bE
            - p.wIP*p.wPI*p.wSS/bE + p.wIP*p.wPS*p.wSI/bE
            + p.wIS*p.wPI*p.wSP/bE - p.wIS*p.wPP*p.wSI/bE
            + p.wII*p.wPP/(bE*bS) - p.wIP*p.wPI/(bE*bS)
            + p.wII*p.wSS/(bE*bP) - p.wIS*p.wSI/(bE*bP)
            + p.wII/(bE*bP*bS)
            + p.wPP*p.wSS/(bE*bI) - p.wPS*p.wSP/(bE*bI)
            + p.wPP/(bE*bI*bS) + p.wSS/(bE*bI*bP) + 1/(bE*bI*bP*bS)
        )

        L_save[0, i] = (
            (-p.wEI*p.wPP*p.wSS + p.wEI*p.wPS*p.wSP + p.wEP*p.wPI*p.wSS
            - p.wEP*p.wPS*p.wSI - wES*p.wPI*p.wSP + wES*p.wPP*p.wSI
            + p.wPP*p.wSS*wEI_correction - p.wPS*p.wSP*wEI_correction
            - p.wEI*p.wPP/bS + p.wEP*p.wPI/bS + p.wPP*wEI_correction/bS
            - p.wEI*p.wSS/bP + wES*p.wSI/bP + p.wSS*wEI_correction/bP
            - p.wEI/(bP*bS) + wEI_correction/(bP*bS)) / det_num
        )
        linear_delta_r[0, i] = L_save[0, i] * p.I_mod_I

        if not only_excitatory:
            L_save[1, i] = (
                (p.wEE*p.wPI*p.wSS - p.wEE*p.wPS*p.wSI - p.wEI*p.wPE*p.wSS
                + p.wEI*p.wPS*p.wSE + wES*p.wPE*p.wSI - wES*p.wPI*p.wSE
                + p.wPE*p.wSS*wEI_correction - p.wPS*p.wSE*wEI_correction
                + p.wEE*p.wPI/bS - p.wEI*p.wPE/bS + p.wPE*wEI_correction/bS
                - p.wPI*p.wSS/bE + p.wPS*p.wSI/bE - p.wPI/(bE*bS)) / det_num
            )
            L_save[2, i] = (
                (-p.wEE*p.wPI*p.wSP + p.wEE*p.wPP*p.wSI + p.wEI*p.wPE*p.wSP
                - p.wEI*p.wPP*p.wSE - p.wEP*p.wPE*p.wSI + p.wEP*p.wPI*p.wSE
                - p.wPE*p.wSP*wEI_correction + p.wPP*p.wSE*wEI_correction
                + p.wEE*p.wSI/bP - p.wEI*p.wSE/bP + p.wSE*wEI_correction/bP
                + p.wPI*p.wSP/bE - p.wPP*p.wSI/bE - p.wSI/(bE*bP)) / det_num
            )
            L_save[3, i] = (
                (-p.wEE*p.wPP*p.wSS + p.wEE*p.wPS*p.wSP + p.wEP*p.wPE*p.wSS
                - p.wEP*p.wPS*p.wSE - wES*p.wPE*p.wSP + wES*p.wPP*p.wSE
                - p.wEE*p.wPP/bS + p.wEP*p.wPE/bS - p.wEE*p.wSS/bP + wES*p.wSE/bP
                - p.wEE/(bP*bS) + p.wPP*p.wSS/bE - p.wPS*p.wSP/bE
                + p.wPP/(bE*bS) + p.wSS/(bE*bP) + 1/(bE*bP*bS)) / det_num
            )
            
            linear_delta_r[1, i] = L_save[1, i] * p.I_mod_I
            linear_delta_r[2, i] = L_save[2, i] * p.I_mod_I
            linear_delta_r[3, i] = L_save[3, i] * p.I_mod_I

    return linear_delta_r
