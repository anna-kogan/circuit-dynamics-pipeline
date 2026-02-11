import numpy as np
from pathlib import Path

from circuit_pipeline.simulation.sweeps import sweep_rP0_delta_rE, sweep_alpha_delta_rates

class ExtendedCircuit:
    def __init__(
            self,
            interneuron_name,
            W,
            tau=np.array([10,10,10,10]),
            rS0=2,
            rI0=2,
            step_rX=0.01,
            I_stim_E=0.3,
            I_stim_P=0.3,
            I_mod_I=0.3,
            power=2,
            mult_f=1/4
    ):
        # Define parameters
        self.interneuron_name = interneuron_name
        self.W = W

        (self.wEE, self.wEP, self.wES_0, self.wEI,
         self.wPE, self.wPP, self.wPS, self.wPI,
         self.wSE, self.wSP, self.wSS, self.wSI,
         self.wIE, self.wIP, self.wIS, self.wII) = W.flatten()
        self.wES = self.wES_0

        self.tauE, self.tauP, self.tauS, self.tauI = tau 

        self.rS0 = rS0  # SST initial firing rate
        self.rI0 = rI0  # Interneuron initial firing rate
        self.step_rX = step_rX

        self.I_stim_E = I_stim_E
        self.I_stim_P = I_stim_P
        self.I_mod_I = I_mod_I

        # I/O function parameters
        self.power = power
        self.mult_f = mult_f

    def _sim_base_kwargs(self, *, mod_onset_t: float = 50.0) -> dict:
        """
        Build the required keyword args for simulate_dynamics/sweeps from this circuit instance.
        """
        tau = np.array([self.tauE, self.tauP, self.tauS, self.tauI], dtype=float)
        return dict(
            W=self.W,
            interneuron_name=self.interneuron_name,
            wES_0=self.wES_0,
            tau=tau,
            rS0=self.rS0,
            rI0=self.rI0,
            I_mod_I=self.I_mod_I,
            power=self.power,
            mult_f=self.mult_f,
            mod_onset_t=float(mod_onset_t)
        )    

##################

    def calculate_linear(
        self,
        max_rEP,
        alfa,
        *,
        save_npz_path: str | Path | None = None,
        force: bool = False,
        progress: bool = False
    ):
        """
        Compatibility wrapper that calls the pipeline grid runner.
        """
        from circuit_pipeline.pipeline.run_grid import run_linear_grid
        from circuit_pipeline.model.linear_response import CircuitParams

        params = CircuitParams(
            interneuron_name=self.interneuron_name,
            W=self.W,
            tau=np.array([self.tauE, self.tauP, self.tauS, self.tauI], dtype=float),
            rS0=self.rS0,
            rI0=self.rI0,
            step_rX=self.step_rX,
            I_stim_E=self.I_stim_E,
            I_stim_P=self.I_stim_P,
            I_mod_I=self.I_mod_I,
            power=self.power,
            mult_f=self.mult_f
        )

        grid = run_linear_grid(
            params=params,
            max_rEP=float(max_rEP),
            step_rX=float(self.step_rX),
            alfa=float(alfa),
            save_npz_path=None if save_npz_path is None else Path(save_npz_path),
            force=bool(force),
            progress=bool(progress)
        )

        return (
            grid.f_gain_num,
            grid.f_gain_num_mod,
            grid.f_maxEVs_num,
            grid.f_maxEVs_num_mod,
            grid.f_maxImEVs_num,
            grid.f_oscMetric_num,
            grid.f_oscMetric_num_mod,
            grid.f_modI_rE_num,
            grid.f_modI_rP_num,
            grid.f_modI_rS_num,
            grid.f_modI_rI_num,
            grid.rE_vec,
            grid.rP_vec
        )

##################

    def plot_heatmaps_with_arrows_and_scatter(
        self,
        *,
        rE_vec,
        rP_vec,
        gain,
        output_path,
        max_ev=None,
        osc_metric=None,
        title=None,
        dpi=150,
        close=True
    ) -> None:
        from circuit_pipeline.plotting.heatmaps import GridPlotData, plot_grid_heatmaps

        data = GridPlotData(
            rE_vec=np.asarray(rE_vec, dtype=float),
            rP_vec=np.asarray(rP_vec, dtype=float),
            gain=np.asarray(gain, dtype=float),
            max_ev=None if max_ev is None else np.asarray(max_ev, dtype=float),
            osc_metric=None if osc_metric is None else np.asarray(osc_metric, dtype=float),
        )
        plot_grid_heatmaps(
            data,
            output_path=Path(output_path),
            title=title,
            dpi=dpi,
            close=close
        )

    
########################

    def calculate_dynamics(
            self,
            rE0,
            rP0,
            alfa,
            dt,
            end_sim,
            mod_onset_t=50.0,
            OnlyDeltas=False
    ):
        """
        Calculates the real dynamics of the system with fixed SST and NDNF/VIP rates and passed E and PV rates.
        Also can calculate only changes of steady state rates before and after NDNF/VIP modulation.
        """
        from circuit_pipeline.simulation.dynamics import simulate_dynamics

        tau = np.array([self.tauE, self.tauP, self.tauS, self.tauI], dtype=float)

        return simulate_dynamics(
            W=self.W,
            interneuron_name=self.interneuron_name,
            wES_0=self.wES_0,
            tau=tau,
            rS0=self.rS0,
            rI0=self.rI0,
            I_mod_I=self.I_mod_I,
            power=self.power,
            mult_f=self.mult_f,
            rE0=rE0,
            rP0=rP0,
            alfa=alfa,
            dt=dt,
            end_sim=end_sim,
            mod_onset_t=mod_onset_t,
            only_deltas=OnlyDeltas
        )
    
#####################
        
    def plot_rate_weight_dynamics(
        self,
        rE0,
        rP0,
        alfa,
        *,
        output_path,
        dt=0.01,
        end_sim=350,
        dpi=150,
        close=True
    ) -> None:
        
        """
        Plots the real dynamics of rates and weights for particular system
        """
        from circuit_pipeline.plotting.dynamics_plots import plot_rate_weight_dynamics as _plot_rate_weight_dynamics

        rE_save, rP_save, rS_save, rI_save, wES_save, wIS_save, _delta_r_vec = self.calculate_dynamics(rE0, rP0, alfa, dt, end_sim)
        time = np.arange(0, end_sim+dt, dt)

        _plot_rate_weight_dynamics(
            time=time,
            rE=rE_save,
            rP=rP_save,
            rS=rS_save,
            rI=rI_save,
            wES=wES_save,
            wIS=wIS_save,
            alfa=alfa,
            end_sim=end_sim,
            output_path=Path(output_path),
            dpi=dpi,
            close=close
        )
    
#############################

    def plot_drE_rP0_fix_alpha(
        self,
        alfa,
        rE0,
        *,
        output_path,
        title,
        rP0_vec=np.arange(0, 201, 1),
        dt=0.01,
        end_sim=350,
        progress=True,
        dpi=150,
        close=True
    ) -> None:
        
        """
        Plots excitatory neuron population rate change against initial rate of PV value
        for fixed alpha and initial rate of E value
        """
        from circuit_pipeline.plotting.sweeps import plot_drE_vs_rP0 as _plot_drE_vs_rP0

        base = self._sim_base_kwargs(mod_onset_t=50.0)

        delta_rE_save = sweep_rP0_delta_rE(
            **base,
            rE0=float(rE0),
            alfa=float(alfa),
            rP0_vec=np.asarray(rP0_vec, dtype=float),
            dt=float(dt),
            end_sim=float(end_sim),
            progress=bool(progress)
        )

        _plot_drE_vs_rP0(
            rP0_vec=np.asarray(rP0_vec, dtype=float),
            delta_rE=np.asarray(delta_rE_save, dtype=float),
            title=title,
            output_path=Path(output_path),
            dpi=dpi,
            close=close
        )
    
#############################     

    def plot_rate_change_alfa(
        self,
        rE0,
        rP0,
        alfa_vec,
        *,
        output_path,
        title,
        dt=0.01,
        end_sim=350,
        LinearCompare=False,
        OnlyExcitatory=True,
        margins=0.1,
        progress=True,
        dpi=150,
        close=True
    ) -> None:
        """
        Plots the rate changes against alpha and saves the figure.
        Uses simulation.sweeps + plotting.sweeps.
        """
        from circuit_pipeline.plotting.sweeps import plot_delta_rates_vs_alpha as _plot_delta_rates_vs_alpha
        from circuit_pipeline.model.linear_compare import LinearCompareParams, compute_linear_delta_over_alpha

        alfa_vec = np.asarray(alfa_vec, dtype=float)

        base = self._sim_base_kwargs(mod_onset_t=50.0)

        delta_r_save = sweep_alpha_delta_rates(
            **base,
            rE0=float(rE0),
            rP0=float(rP0),
            alfa_vec=alfa_vec,
            dt=float(dt),
            end_sim=float(end_sim),
            progress=bool(progress)
        )

        # optional linear compare
        linear_delta_r_save = None
        if LinearCompare:
            p = LinearCompareParams(
                interneuron_name=self.interneuron_name,

                wEE=self.wEE, wEP=self.wEP, wES_0=self.wES_0, wEI=self.wEI,
                wPE=self.wPE, wPP=self.wPP, wPS=self.wPS, wPI=self.wPI,
                wSE=self.wSE, wSP=self.wSP, wSS=self.wSS, wSI=self.wSI,
                wIE=self.wIE, wIP=self.wIP, wIS=self.wIS, wII=self.wII,

                rS0=self.rS0,
                rI0=self.rI0,

                power=self.power,
                mult_f=self.mult_f,
                I_mod_I=self.I_mod_I
            )

            linear_delta_r_save = compute_linear_delta_over_alpha(
                p,
                rE0=float(rE0),
                rP0=float(rP0),
                alfa_vec=alfa_vec,
                only_excitatory=bool(OnlyExcitatory)
            )

        # 3) plot + save using plotting module
        _plot_delta_rates_vs_alpha(
            alfa_vec=alfa_vec,
            delta_r=delta_r_save,
            title=title,
            output_path=Path(output_path),
            only_excitatory=OnlyExcitatory,
            linear_compare=LinearCompare,
            linear_delta_r=linear_delta_r_save,
            margins=margins,
            dpi=dpi,
            close=close
        )
    
#######################################

    def plot_rates_for_alphas(
        self,
        rE0,
        rP0,
        alfa_vec,
        *,
        output_path,
        dt=0.01,
        end_sim=350,
        dpi=150,
        close=True
    ) -> None:
        """
        Saves rate dynamics for several different alphas as a subplot grid.
        """
        from circuit_pipeline.plotting.rates_grid import save_rates_grid_for_alphas

        alfa_vec = np.asarray(alfa_vec, dtype=float)
        time = np.arange(0, float(end_sim) + float(dt), float(dt), dtype=float)

        rates_by_alpha: list[dict[str, np.ndarray]] = []

        for alfa in alfa_vec:
            res = self.calculate_dynamics(rE0, rP0, float(alfa), float(dt), float(end_sim))

            # calculate_dynamics returns full trajectories when OnlyDeltas=False
            rE_save, rP_save, rS_save, rI_save, *_ = res

            rates_by_alpha.append(
                dict(
                    rE=np.asarray(rE_save, dtype=float),
                    rP=np.asarray(rP_save, dtype=float),
                    rS=np.asarray(rS_save, dtype=float),
                    rI=np.asarray(rI_save, dtype=float)
                )
            )

        save_rates_grid_for_alphas(
            time=time,
            rates_by_alpha=rates_by_alpha,
            alfa_vec=alfa_vec,
            output_path=Path(output_path),
            dpi=dpi,
            close=close
        )

        
