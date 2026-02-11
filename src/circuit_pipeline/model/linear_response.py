from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.linalg import eig


@dataclass(frozen=True)
class CircuitParams:
    interneuron_name: str
    W: np.ndarray            # shape (4,4)
    tau: np.ndarray          # shape (4,)
    rS0: float
    rI0: float
    step_rX: float

    I_stim_E: float
    I_stim_P: float
    I_mod_I: float

    power: float
    mult_f: float


@dataclass(frozen=True)
class PointResult:
    # pre-mod
    gain: float
    max_ev: float
    max_im_ev: float
    osc_metric: float

    # modulation deltas
    modI_rE: float
    modI_rP: float
    modI_rS: float
    modI_rI: float

    # post-mod
    gain_mod: float
    max_ev_mod: float
    osc_metric_mod: float

    # flags
    det_singular: bool
    det_singular_mod: bool
    eig_failed: bool
    eig_failed_mod: bool


def _unpack_W(W: np.ndarray):
    (wEE, wEP, wES_0, wEI,
     wPE, wPP, wPS,  wPI,
     wSE, wSP, wSS,  wSI,
     wIE, wIP, wIS,  wII) = W.flatten()
    return (wEE, wEP, wES_0, wEI, wPE, wPP, wPS, wPI, wSE, wSP, wSS, wSI, wIE, wIP, wIS, wII)


def compute_point(params: CircuitParams, rE: float, rP: float, alfa: float) -> PointResult:
    """
    Compute all quantities for a single grid point (rE, rP) (at a fixed alfa for NDNF).
    """

    W = np.asarray(params.W, dtype=float)
    tau = np.asarray(params.tau, dtype=float)
    tauE, tauP, tauS, tauI = tau

    rS = float(params.rS0)
    rI = float(params.rI0)

    power = float(params.power)
    mult_f = float(params.mult_f)

    I_stim_E = float(params.I_stim_E)
    I_stim_P = float(params.I_stim_P)
    I_mod_I = float(params.I_mod_I)

    (wEE, wEP, wES_0, wEI,
     wPE, wPP, wPS, wPI,
     wSE, wSP, wSS, wSI,
     wIE, wIP, wIS, wII) = _unpack_W(W)

    # ---------- BEFORE modulation ----------
    # weight dependence correction
    if params.interneuron_name == "ndnf":
        wES_mod = wES_0 * np.exp(-rI / alfa)
        pESN = (wES_0 / alfa) * np.exp(-rI / alfa)
        wEI_correction = pESN * rS
        if wEI_correction > wEI:
            wEI_correction = wEI
    elif params.interneuron_name == "vip":
        wES_mod = wES_0
        wEI_correction = 0.0
    else:
        raise ValueError('False name of the interneuron. Available options: "vip" or "ndnf".')

    # inputs
    xE = (rE / mult_f) ** (1 / power) - (wEE * rE - wEP * rP - wES_mod * rS - wEI * rI)
    xP = (rP / mult_f) ** (1 / power) - (wPE * rE - wPP * rP - wPS * rS - wPI * rI)
    xS = (rS / mult_f) ** (1 / power) - (wSE * rE - wSP * rP - wSS * rS - wSI * rI)
    xI = (rI / mult_f) ** (1 / power) - (wIE * rE - wIP * rP - wIS * rS - wII * rI)

    # gains bE..bI
    bE = power * mult_f * (wEE * rE - wEP * rP - wES_mod * rS - wEI * rI + xE) ** (power - 1)
    bP = power * mult_f * (wPE * rE - wPP * rP - wPS * rS - wPI * rI + xP) ** (power - 1)
    bS = power * mult_f * (wSE * rE - wSP * rP - wSS * rS - wSI * rI + xS) ** (power - 1)
    bI = power * mult_f * (wIE * rE - wIP * rP - wIS * rS - wII * rI + xI) ** (power - 1)

    # determinant before modulation
    det_num = (
        -wEE*wII*wPP*wSS + wEE*wII*wPS*wSP + wEE*wIP*wPI*wSS - wEE*wIP*wPS*wSI - wEE*wIS*wPI*wSP + wEE*wIS*wPP*wSI + wEI*wIE*wPP*wSS - wEI*wIE*wPS*wSP
        -wEI*wIP*wPE*wSS + wEI*wIP*wPS*wSE + wEI*wIS*wPE*wSP - wEI*wIS*wPP*wSE - wEP*wIE*wPI*wSS + wEP*wIE*wPS*wSI + wEP*wII*wPE*wSS - wEP*wII*wPS*wSE
        -wEP*wIS*wPE*wSI + wEP*wIS*wPI*wSE + wES_mod*wIE*wPI*wSP - wES_mod*wIE*wPP*wSI - wES_mod*wII*wPE*wSP + wES_mod*wII*wPP*wSE + wES_mod*wIP*wPE*wSI - wES_mod*wIP*wPI*wSE
        -wIE*wPP*wSS*wEI_correction + wIE*wPS*wSP*wEI_correction + wIP*wPE*wSS*wEI_correction - wIP*wPS*wSE*wEI_correction - wIS*wPE*wSP*wEI_correction
        +wIS*wPP*wSE*wEI_correction - wEE*wII*wPP/bS + wEE*wIP*wPI/bS + wEI*wIE*wPP/bS - wEI*wIP*wPE/bS - wEP*wIE*wPI/bS + wEP*wII*wPE/bS
        -wIE*wPP*wEI_correction/bS + wIP*wPE*wEI_correction/bS - wEE*wII*wSS/bP + wEE*wIS*wSI/bP + wEI*wIE*wSS/bP - wEI*wIS*wSE/bP - wES_mod*wIE*wSI/bP
        +wES_mod*wII*wSE/bP - wIE*wSS*wEI_correction/bP + wIS*wSE*wEI_correction/bP - wEE*wII/(bP*bS) + wEI*wIE/(bP*bS) - wIE*wEI_correction/(bP*bS)
        -wEE*wPP*wSS/bI + wEE*wPS*wSP/bI + wEP*wPE*wSS/bI - wEP*wPS*wSE/bI - wES_mod*wPE*wSP/bI + wES_mod*wPP*wSE/bI - wEE*wPP/(bI*bS) + wEP*wPE/(bI*bS)
        -wEE*wSS/(bI*bP) + wES_mod*wSE/(bI*bP) - wEE/(bI*bP*bS) + wII*wPP*wSS/bE - wII*wPS*wSP/bE - wIP*wPI*wSS/bE + wIP*wPS*wSI/bE + wIS*wPI*wSP/bE
        -wIS*wPP*wSI/bE + wII*wPP/(bE*bS) - wIP*wPI/(bE*bS) + wII*wSS/(bE*bP) - wIS*wSI/(bE*bP) + wII/(bE*bP*bS) + wPP*wSS/(bE*bI) - wPS*wSP/(bE*bI)
        +wPP/(bE*bI*bS) + wSS/(bE*bI*bP) + 1/(bE*bI*bP*bS)
    )

    det_singular = bool(abs(det_num) < 1e-10)
    if det_singular:
        nan = float("nan")
        return PointResult(
            gain=nan, max_ev=nan, max_im_ev=nan, osc_metric=nan,
            modI_rE=nan, modI_rP=nan, modI_rS=nan, modI_rI=nan,
            gain_mod=nan, max_ev_mod=nan, osc_metric_mod=nan,
            det_singular=True, det_singular_mod=True,
            eig_failed=True, eig_failed_mod=True
        )

    # response matrix L terms
    L_EE = (
        (wII*wPP*wSS - wII*wPS*wSP - wIP*wPI*wSS + wIP*wPS*wSI + wIS*wPI*wSP - wIS*wPP*wSI + wII*wPP/bS - wIP*wPI/bS + wII*wSS/bP - wIS*wSI/bP
        +wII/(bP*bS) + wPP*wSS/bI - wPS*wSP/bI + wPP/(bI*bS) + wSS/(bI*bP) + 1/(bI*bP*bS)) / det_num
    )
    L_EP = (
        (wEI*wIP*wSS - wEI*wIS*wSP - wEP*wII*wSS + wEP*wIS*wSI + wES_mod*wII*wSP - wES_mod*wIP*wSI - wIP*wSS*wEI_correction + wIS*wSP*wEI_correction
        +wEI*wIP/bS - wEP*wII/bS - wIP*wEI_correction/bS - wEP*wSS/bI + wES_mod*wSP/bI - wEP/(bI*bS)) / det_num
    )
    L_EI = (
        (-wEI*wPP*wSS + wEI*wPS*wSP + wEP*wPI*wSS - wEP*wPS*wSI - wES_mod*wPI*wSP + wES_mod*wPP*wSI + wPP*wSS*wEI_correction - wPS*wSP*wEI_correction - wEI*wPP/bS
        +wEP*wPI/bS + wPP*wEI_correction/bS - wEI*wSS/bP + wES_mod*wSI/bP + wSS*wEI_correction/bP - wEI/(bP*bS) + wEI_correction/(bP*bS)) / det_num
    )
    L_PI = (
        (wEE*wPI*wSS - wEE*wPS*wSI - wEI*wPE*wSS + wEI*wPS*wSE + wES_mod*wPE*wSI - wES_mod*wPI*wSE + wPE*wSS*wEI_correction - wPS*wSE*wEI_correction + wEE*wPI/bS
        -wEI*wPE/bS + wPE*wEI_correction/bS - wPI*wSS/bE + wPS*wSI/bE - wPI/(bE*bS)) / det_num
    )
    L_SI = (
        (-wEE*wPI*wSP + wEE*wPP*wSI + wEI*wPE*wSP - wEI*wPP*wSE - wEP*wPE*wSI + wEP*wPI*wSE - wPE*wSP*wEI_correction + wPP*wSE*wEI_correction
        +wEE*wSI/bP - wEI*wSE/bP + wSE*wEI_correction/bP + wPI*wSP/bE - wPP*wSI/bE - wSI/(bE*bP)) / det_num
    )
    L_II = (
        (-wEE*wPP*wSS + wEE*wPS*wSP + wEP*wPE*wSS - wEP*wPS*wSE - wES_mod*wPE*wSP + wES_mod*wPP*wSE - wEE*wPP/bS + wEP*wPE/bS - wEE*wSS/bP + wES_mod*wSE/bP
        -wEE/(bP*bS) + wPP*wSS/bE - wPS*wSP/bE + wPP/(bE*bS) + wSS/(bE*bP) + 1/(bE*bP*bS)) / det_num
    )

    gain = L_EE * I_stim_E + L_EP * I_stim_P

    # Jacobian
    J_num = np.array([
        [(bE*wEE - 1)/tauE, -bE*wEP/tauE, -bE*wES_mod/tauE, bE*(-wEI + wEI_correction)/tauE],
        [bP*wPE/tauP, (-bP*wPP - 1)/tauP, -bP*wPS/tauP, -bP*wPI/tauP],
        [bS*wSE/tauS, -bS*wSP/tauS, (-bS*wSS - 1)/tauS, -bS*wSI/tauS],
        [bI*wIE/tauI, -bI*wIP/tauI, -bI*wIS/tauI, (-bI*wII - 1)/tauI]
    ], dtype=float)

    eig_failed = False
    try:
        ev = eig(J_num)[0]
        max_ev = float(np.max(np.real(ev)))
        max_im_ev = float(np.max(np.imag(ev)))
        idx = int(np.argmax(np.real(ev)))
        im2 = float(np.imag(ev[idx])**2)
        re2 = float(max_ev**2)
        osc_metric = float(im2 / (im2 + re2)) if (im2 + re2) != 0 else 0.0
    except Exception:
        eig_failed = True
        max_ev = float("nan")
        max_im_ev = float("nan")
        osc_metric = float("nan")

    # modulation deltas
    modI_rE = float(L_EI * I_mod_I)
    modI_rP = float(L_PI * I_mod_I)
    modI_rS = float(L_SI * I_mod_I)
    modI_rI = float(L_II * I_mod_I)

    # ---------- AFTER modulation ----------
    rS_mod = rS + modI_rS
    rI_mod = rI + modI_rI

    if params.interneuron_name == "ndnf":
        wES_mod = wES_0 * np.exp(-rI_mod / alfa)
        pESN_mod = (wES_0 / alfa) * np.exp(-rI_mod / alfa)
        wEI_correction_mod = pESN_mod * rS
        if wEI_correction_mod > wEI:
            wEI_correction_mod = wEI
    else:
        wES_mod = wES_0
        wEI_correction_mod = 0.0

    # gains after modulation
    bE_mod = power * mult_f * (wEE*(rE + modI_rE) - wEP*(rP + modI_rP) - wES_mod*(rS + modI_rS) - wEI*(rI + modI_rI) + xE)**(power-1)
    bP_mod = power * mult_f * (wPE*(rE + modI_rE) - wPP*(rP + modI_rP) - wPS*(rS + modI_rS) - wPI*(rI + modI_rI) + xP)**(power-1)
    bS_mod = power * mult_f * (wSE*(rE + modI_rE) - wSP*(rP + modI_rP) - wSS*(rS + modI_rS) - wSI*(rI + modI_rI) + xS)**(power-1)
    bI_mod = power * mult_f * (wIE*(rE + modI_rE) - wIP*(rP + modI_rP) - wIS*(rS + modI_rS) - wII*(rI + modI_rI) + xI + I_mod_I)**(power-1)

    # determinant after modulation
    det_num_mod = (
        -wEE*wII*wPP*wSS + wEE*wII*wPS*wSP + wEE*wIP*wPI*wSS - wEE*wIP*wPS*wSI - wEE*wIS*wPI*wSP + wEE*wIS*wPP*wSI + wEI*wIE*wPP*wSS - wEI*wIE*wPS*wSP
        -wEI*wIP*wPE*wSS + wEI*wIP*wPS*wSE + wEI*wIS*wPE*wSP - wEI*wIS*wPP*wSE - wEP*wIE*wPI*wSS + wEP*wIE*wPS*wSI + wEP*wII*wPE*wSS - wEP*wII*wPS*wSE
        -wEP*wIS*wPE*wSI + wEP*wIS*wPI*wSE + wES_mod*wIE*wPI*wSP - wES_mod*wIE*wPP*wSI - wES_mod*wII*wPE*wSP + wES_mod*wII*wPP*wSE + wES_mod*wIP*wPE*wSI
        -wES_mod*wIP*wPI*wSE - wIE*wPP*wSS*wEI_correction_mod + wIE*wPS*wSP*wEI_correction_mod + wIP*wPE*wSS*wEI_correction_mod
        -wIP*wPS*wSE*wEI_correction_mod - wIS*wPE*wSP*wEI_correction_mod + wIS*wPP*wSE*wEI_correction_mod - wEE*wII*wPP/bS_mod + wEE*wIP*wPI/bS_mod
        +wEI*wIE*wPP/bS_mod - wEI*wIP*wPE/bS_mod - wEP*wIE*wPI/bS_mod + wEP*wII*wPE/bS_mod - wIE*wPP*wEI_correction_mod/bS_mod
        +wIP*wPE*wEI_correction_mod/bS_mod - wEE*wII*wSS/bP_mod + wEE*wIS*wSI/bP_mod + wEI*wIE*wSS/bP_mod - wEI*wIS*wSE/bP_mod - wES_mod*wIE*wSI/bP_mod
        +wES_mod*wII*wSE/bP_mod - wIE*wSS*wEI_correction_mod/bP_mod + wIS*wSE*wEI_correction_mod/bP_mod - wEE*wII/(bP_mod*bS_mod)
        +wEI*wIE/(bP_mod*bS_mod) - wIE*wEI_correction_mod/(bP_mod*bS_mod) - wEE*wPP*wSS/bI_mod + wEE*wPS*wSP/bI_mod + wEP*wPE*wSS/bI_mod
        -wEP*wPS*wSE/bI_mod - wES_mod*wPE*wSP/bI_mod + wES_mod*wPP*wSE/bI_mod - wEE*wPP/(bI_mod*bS_mod) + wEP*wPE/(bI_mod*bS_mod)
        -wEE*wSS/(bI_mod*bP_mod) + wES_mod*wSE/(bI_mod*bP_mod) - wEE/(bI_mod*bP_mod*bS_mod) + wII*wPP*wSS/bE_mod - wII*wPS*wSP/bE_mod - wIP*wPI*wSS/bE_mod
        +wIP*wPS*wSI/bE_mod + wIS*wPI*wSP/bE_mod - wIS*wPP*wSI/bE_mod + wII*wPP/(bE_mod*bS_mod) - wIP*wPI/(bE_mod*bS_mod) + wII*wSS/(bE_mod*bP_mod)
        -wIS*wSI/(bE_mod*bP_mod) + wII/(bE_mod*bP_mod*bS_mod) + wPP*wSS/(bE_mod*bI_mod) - wPS*wSP/(bE_mod*bI_mod) + wPP/(bE_mod*bI_mod*bS_mod)
        +wSS/(bE_mod*bI_mod*bP_mod) + 1/(bE_mod*bI_mod*bP_mod*bS_mod)
    )

    det_singular_mod = bool(abs(det_num_mod) < 1e-10)
    if det_singular_mod:
        return PointResult(
            gain=float(gain), max_ev=float(max_ev), max_im_ev=float(max_im_ev), osc_metric=float(osc_metric),
            modI_rE=modI_rE, modI_rP=modI_rP, modI_rS=modI_rS, modI_rI=modI_rI,
            gain_mod=float("nan"), max_ev_mod=float("nan"), osc_metric_mod=float("nan"),
            det_singular=False, det_singular_mod=True,
            eig_failed=eig_failed, eig_failed_mod=True
        )

    # Response matrix terms L_EE_mod, L_EP_mod and Jacobian after modulation
    L_EE_mod = (
        (wII*wPP*wSS - wII*wPS*wSP - wIP*wPI*wSS + wIP*wPS*wSI + wIS*wPI*wSP - wIS*wPP*wSI + wII*wPP/bS_mod - wIP*wPI/bS_mod
         +wII*wSS/bP_mod - wIS*wSI/bP_mod + wII/(bP_mod*bS_mod) + wPP*wSS/bI_mod - wPS*wSP/bI_mod + wPP/(bI_mod*bS_mod)
         +wSS/(bI_mod*bP_mod) + 1/(bI_mod*bP_mod*bS_mod)) / det_num_mod
    )
    L_EP_mod = (
        (wEI*wIP*wSS - wEI*wIS*wSP - wEP*wII*wSS + wEP*wIS*wSI + wES_mod*wII*wSP - wES_mod*wIP*wSI - wIP*wSS*wEI_correction_mod + wIS*wSP*wEI_correction_mod
         +wEI*wIP/bS_mod - wEP*wII/bS_mod - wIP*wEI_correction_mod/bS_mod - wEP*wSS/bI_mod + wES_mod*wSP/bI_mod - wEP/(bI_mod*bS_mod)) / det_num_mod
    )

    gain_mod = L_EE_mod * I_stim_E + L_EP_mod * I_stim_P

    J_num_mod = np.array([
        [(bE_mod*wEE - 1)/tauE, -bE_mod*wEP/tauE, -bE_mod*wES_mod/tauE, bE_mod*(-wEI + wEI_correction_mod)/tauE],
        [bP_mod*wPE/tauP, (-bP_mod*wPP - 1)/tauP, -bP_mod*wPS/tauP, -bP_mod*wPI/tauP],
        [bS_mod*wSE/tauS, -bS_mod*wSP/tauS, (-bS_mod*wSS - 1)/tauS, -bS_mod*wSI/tauS],
        [bI_mod*wIE/tauI, -bI_mod*wIP/tauI, -bI_mod*wIS/tauI, (-bI_mod*wII - 1)/tauI]
    ])

    eig_failed_mod = False
    try:
        ev_mod = eig(J_num_mod)[0]
        max_ev_mod = float(np.max(np.real(ev_mod)))
        idx_mod = int(np.argmax(np.real(ev_mod)))
        im2_mod = float(np.imag(ev_mod[idx_mod])**2)
        re2_mod = float(max_ev_mod**2)
        osc_metric_mod = float(im2_mod / (im2_mod + re2_mod)) if (im2_mod + re2_mod) != 0 else 0.0
    except Exception:
        eig_failed_mod = True
        max_ev_mod = float("nan")
        osc_metric_mod = float("nan")

    return PointResult(
        gain=float(gain), max_ev=float(max_ev), max_im_ev=float(max_im_ev), osc_metric=float(osc_metric),
        modI_rE=modI_rE, modI_rP=modI_rP, modI_rS=modI_rS, modI_rI=modI_rI,
        gain_mod=float(gain_mod), max_ev_mod=float(max_ev_mod), osc_metric_mod=float(osc_metric_mod),
        det_singular=False, det_singular_mod=False,
        eig_failed=eig_failed, eig_failed_mod=eig_failed_mod
    )
