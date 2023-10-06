import numpy as np

from src.environment.simulations.optimisation_post_processing import \
    check_temp_equations
import src.environment.utilities.userdeftools as utils


class ExplorerTests:
    def __init__(self, explorer):
        self.explorer = explorer
        for attribute in ['data', 'prm', 'paths', 'N', 'grd', 'n_homes', 'rl']:
            setattr(self, attribute, getattr(explorer, attribute))

    def investigate_opt_env_rewards_unequal(self, step_vals, evaluation):
        # check env batch matches res
        res = np.load(
            self.paths['opt_res'] / self.data.get_res_name(evaluation),
            allow_pickle=True
        ).item()
        if abs(np.sum(step_vals['opt']['reward']) + np.sum(res['hourly_total_costs'])) > 1e-3:
            print(
                f"investigate_opt_env_rewards_unequal: opt reward {np.sum(step_vals['opt']['reward'])} should correspond to "
                f"res total costs {np.sum(res['hourly_total_costs'])}"
            )
        if not (
                abs(
                    res['discharge_tot'] - res['discharge_other'] / self.prm['car']['eta_dis']
                    - self.explorer.env.batch['loads_car'][:, 0: self.N]
                ) < 1e-3
        ).all():
            print("investigate_opt_env_rewards_unequal: es charge/discharge to not match batch loads_car")

        # check total household consumption matches between res and env
        if abs(np.sum(self.explorer.env.tot_cons_loads) - np.sum(res['totcons'] - res['E_heat'])) > 1e-3:
            print(
                f"investigate_opt_env_rewards_unequal: tot_cons_loads env {np.sum(self.explorer.env.tot_cons_loads)} "
                f"and res {np.sum(res['totcons'] - res['E_heat'])} not matching"
            )

        # check energy prices used are the same
        if sum(
            self.grd['C_test'][time_step] * (res['grid'][time_step] + self.grd['loss'] * res['grid2'][time_step])
            for time_step in range(self.N)
        ) != res['grid_energy_costs']:
            print("investigate_opt_env_rewards_unequal: grid energy costs not matching")

        # check that heat T_out is the same as in opt
        T_out = self.explorer.env.heat.T_out[0: 24]
        check_temp_equations(res, self.prm['syst'], self.prm['heat'], T_out)

        # check loads_car matches
        for home in range(self.n_homes):
            if abs(
                np.sum(
                    res['discharge_tot'][home] - res['discharge_other'][home] / self.prm['car']['eta_dis']
                )
                - np.sum(self.explorer.env.batch['loads_car'][home, 0: self.N])
            ) > 1e-3:
                print("car charge/discharge not matching env batch loads_car")

    def check_cons_less_than_or_equal_to_available_loads(self, loads, res, time_step, batchflex_opt, evaluation):
        cons_tol = 1e-1
        assert all(
            (loads["l_flex"] + loads["l_fixed"])
            - (res['totcons'][:, time_step] - res['E_heat'][:, time_step])
            >= - cons_tol
        ), f"res loads cons {res['totcons'][:, time_step] - res['E_heat'][:, time_step]}, " \
           f"available loads {loads['l_flex'] + loads['l_fixed']}"

    def tests_individual_step_rl_matches_res(
            self, res, time_step, batch, reward, break_down_rewards, batchflex_opt, evaluation
    ):
        prm = self.prm
        for home in range(self.n_homes):
            fixed_flex = sum(batchflex_opt[home][time_step]) \
                 + self.explorer.env.heat.E_heat_min[home] \
                 + self.explorer.env.heat.potential_E_flex()[home]
            assert res["totcons"][home][time_step] <= fixed_flex + 1e-1, \
                f"cons {res['totcons'][home][time_step]} more than sum fixed + flex" \
                f" {fixed_flex} for home = {home}, time_step = {time_step}"

        # check loads and consumption match
        sum_consa = 0
        for load_type in range(2):
            sum_consa += np.sum(res[f'consa({load_type})'])

        assert len(np.shape(batch['loads'])) == 2, f"np.shape(loads) == {np.shape(batch['loads'])}"
        assert abs((np.sum(batch['loads'][:, 0: self.N]) - sum_consa) / sum_consa) < 1e-2, \
            f"res cons {sum_consa} does not match input demand " \
            f"{np.sum(batch['loads'][:, 0: self.N])}"

        gc_i = prm["grd"][f"C{utils.test_str(evaluation)}"][time_step] * (
                res['grid'][time_step] + prm["grd"]['loss'] * res['grid2'][time_step]
        )
        gc_per_start_i = [
            prm["grd"][f"Call{utils.test_str(evaluation)}"][i + time_step] * (
                    res['grid'][time_step] + prm["grd"]['loss'] * res['grid2'][time_step]
            )
            for i in range(len(prm['grd'][f'Call{utils.test_str(evaluation)}']) - self.N)
        ]
        potential_i0s = [
            i for i, gc_start_i in enumerate(gc_per_start_i)
            if abs(gc_start_i - gc_i) < 1e-3
        ]
        assert self.explorer.env.i0_costs in potential_i0s

        # check reward from environment and res variables match
        if not prm["RL"]["competitive"]:
            if abs(reward - self.rl['delta_reward'] + res['hourly_total_costs'][time_step]) > 5e-3:
                tot_delta = reward + res['hourly_total_costs'][time_step]
                print(
                    f"reward {reward} != "
                    f"res reward {- res['hourly_total_costs'][time_step]} "
                    f"(delta {tot_delta})"
                )
                labels = [
                    'grid_energy_costs',
                    'battery_degradation_costs',
                    'distribution_network_export_costs',
                    'import_export_costs',
                    'voltage_costs'
                ]
                for label in labels:
                    sub_cost_env = break_down_rewards[
                        self.prm['syst']['break_down_rewards_entries'].index(label)
                    ]
                    sub_cost_res = res[f'hourly_{label}'][time_step]
                    if abs(sub_cost_env - sub_cost_res) > 1e-3:
                        sub_delta = sub_cost_env - sub_cost_res
                        print(
                            f"{label} costs do not match: env {sub_cost_env} vs res {sub_cost_res} "
                            f"(error {sub_delta} is {sub_delta / tot_delta * 100} % of total delta)"
                        )

            assert abs(
                reward - self.rl['delta_reward'] + res['hourly_total_costs'][time_step]
            ) < 5e-3, f"reward env {reward} != reward opt {- res['hourly_total_costs'][time_step]}"
    
    def test_total_rewards_match(self, evaluation, res, sum_rl_rewards):
        if not (self.rl["competitive"] and not evaluation):
            assert abs(
                sum_rl_rewards - self.N * self.rl['delta_reward'] + res['total_costs']
            ) < 1e-2, \
                "tot rewards don't match: " \
                f"sum_RL_rewards = {sum_rl_rewards}, " \
                f"sum costs opt = {res['total_costs']}" \
                f"abs(sum_rl_rewards + res['total_costs']) " \
                f"{abs(sum_rl_rewards + res['total_costs'])}"

    def check_competitive_has_diff_rewards(self, diff_rewards):
        if self.rl["competitive"]:
            assert diff_rewards is not None
