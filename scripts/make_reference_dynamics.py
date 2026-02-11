from pathlib import Path
import numpy as np
import yaml

from circuit_pipeline.legacy.extended_circuit import ExtendedCircuit


def main() -> None:
    cfg_path = Path("configs/reference_dynamics.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    c = cfg["circuit"]
    g = cfg["grid"]
    out_npz = Path(cfg["output"]["npz_path"])
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    circuit = ExtendedCircuit(
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

    results = circuit.calculate_linear(
        max_rEP=float(g["max_rEP"]),
        alfa=float(g["alfa"]),
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
    ) = results

    np.savez(
        out_npz,
        cfg_yaml=np.array(cfg_path.read_text(encoding="utf-8"), dtype=object),
        interneuron_name=np.array(c["interneuron_name"], dtype=object),
        W=np.array(c["W"], dtype=float),
        tau=np.array(c["tau"], dtype=float),
        rS0=np.array(float(c["rS0"])),
        rI0=np.array(float(c["rI0"])),
        step_rX=np.array(float(c["step_rX"])),
        I_stim_E=np.array(float(c["I_stim_E"])),
        I_stim_P=np.array(float(c["I_stim_P"])),
        I_mod_I=np.array(float(c["I_mod_I"])),
        power=np.array(float(c["power"])),
        mult_f=np.array(float(c["mult_f"])),
        max_rEP=np.array(float(g["max_rEP"])),
        alfa=np.array(float(g["alfa"])),
        rE_vec=np.array(circuit.rE_vec, dtype=float),
        rP_vec=np.array(circuit.rP_vec, dtype=float),
        f_gain_num=f_gain_num,
        f_gain_num_mod=f_gain_num_mod,
        f_maxEVs_num=f_maxEVs_num,
        f_maxEVs_num_mod=f_maxEVs_num_mod,
        f_maxImEVs_num=f_maxImEVs_num,
        f_oscMetric_num=f_oscMetric_num,
        f_oscMetric_num_mod=f_oscMetric_num_mod,
        f_modI_rE_num=f_modI_rE_num,
        f_modI_rP_num=f_modI_rP_num,
        f_modI_rS_num=f_modI_rS_num,
        f_modI_rI_num=f_modI_rI_num,
    )

    print("Saved:", out_npz)


if __name__ == "__main__":
    main()
