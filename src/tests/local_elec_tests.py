import numpy as np
import pickle


class LocalElecTests:
    def __init__(self, env):
        self.env = env
        for attribute in ['prm', 'N', 'n_homes', 'rl', 'test', 'max_delay']:
            setattr(self, attribute, getattr(env, attribute))

    @property
    def time_step(self):
        return self.env.time_step

    @property
    def batch(self):
        return self.env.batch

    @property
    def ext(self):
        return self.env.ext

    @property
    def tot_cons_loads(self):
        return self.env.tot_cons_loads

    @property
    def car(self):
        return self.env.car
    
    @property
    def batch_flex(self):
        return self.env.batch_flex

    @property
    def d_loaded(self):
        return self.env.d_loaded

    def check_shape_batch_flex(self, batch_flex):
        assert np.shape(batch_flex)[0] == self.n_homes, \
            f"np.shape(batch_flex) {np.shape(batch_flex)} " \
            f"self.n_homes {self.n_homes}"
        assert np.shape(batch_flex)[2] == self.max_delay + 1, \
            f"np.shape(batch_flex) {np.shape(batch_flex)} " \
            f"self.max_delay {self.env.max_delay}"
    
    def check_flex_and_remaining_cons_after_update_home(self, flex_cons, batch_flex, remaining_cons, home, time_step):
        assert flex_cons[home] <= np.sum(batch_flex[home][time_step][1:]) + 5e-2, \
            f"flex_cons[home={home}] {flex_cons[home]} " \
            f"> np.sum(batch_flex[home][time_step={time_step}][1:]) {np.sum(batch_flex[home][time_step][1:])} + 1e-2"
        assert remaining_cons <= 1e-1, \
            f"remaining_cons = {remaining_cons} too large"

    def test_flex_cons(self, time_step, batch_flex):
        if self.test:
            for home in range(self.n_homes):
                lb = np.sum(self.prm['grd']['loads'][0, home, 0: time_step])
                ub = np.sum(self.prm['grd']['loads'][:, home, 0: time_step])
                consumed_so_far = np.sum(np.array(self.tot_cons_loads)[:, home])
                if self.time_step == self.N:
                    consumed_so_far_ok = abs(consumed_so_far - ub) < 1e-3
                else:
                    consumed_so_far_ok = lb - 1e-3 < consumed_so_far < ub + 1e-3
                assert consumed_so_far_ok, f"home {home} time_step {time_step}"
                left_to_consume = np.sum(batch_flex[home, time_step: self.N])
                total_to_consume = np.sum(self.prm['grd']['loads'][:, home, 0: self.N])
                cons_adds_up = abs(left_to_consume + consumed_so_far - total_to_consume) < 1e-3
                assert cons_adds_up, f"home {home} self.time_step {time_step}"

    def check_no_values_issing_break_down_rewards(self, break_down_rewards):
        assert len(break_down_rewards) == len(self.prm['syst']['break_down_rewards_entries']), \
            f"len(break_down_rewards) {len(break_down_rewards)} " \
            f"== len(self.prm['syst']['break_down_rewards_entries']) " \
            f"{len(self.prm['syst']['break_down_rewards_entries'])}"

    @property
    def share_flexs(self):
        return self.env.share_flexs

    def check_no_flex_left_unmet(self, home_vars, loads, h):
        for home in range(self.n_homes):
            assert home_vars['tot_cons'][home] + 1e-3 >= loads['l_fixed'][home], \
                f"home = {home}, no flex cons at last time step"
            assert loads['l_fixed'][home] \
                   >= self.batch['loads'][home, h] * (1 - self.share_flexs[home]), \
                f"home {home} l_fixed and batch do not match"

    def check_time_car_and_env_match(self, time_step):
        assert self.car.time_step == time_step, \
                f"self.car.time_step {self.car.time_step} time_step {time_step}"
    
    def batch_tests(self, time_step):
        if not self.test:
            return
        for home in range(self.n_homes):
            fixed_to_flex = self.share_flexs[home] / (1 - self.share_flexs[home])
            assert sum(self.batch_flex[home][time_step][1: 5]) \
                <= sum(self.batch_flex[home][0: time_step + 1, 0]) * fixed_to_flex, \
                "batch_flex too large time_step"

            assert sum(self.batch_flex[home][time_step + 1][1: 5]) <= sum(
                self.batch_flex[home][ih][0]
                / (1 - self.share_flexs[home]) * self.share_flexs[home]
                for ih in range(0, time_step + 2)), "batch_flex too large time_step + 1"

            n = min(time_step + 30, len(self.batch['loads'][home]))
            for ih in range(time_step, n):
                assert self.batch['loads'][home, ih] \
                    <= self.batch_flex[home][ih][0] \
                    + self.batch_flex[home][ih][-1] + 1e-3, \
                    "loads larger than with flex"

        self.check_batch_flex(time_step)

    def check_flex_not_too_large(self, new_batch_flex, batch_flex, home, time_step):
        if not self.test:
            return
        assert np.sum(new_batch_flex[home][0][1:5]) <= sum(
            batch_flex[home][ih][0] / (1 - self.share_flexs[home])
            * self.share_flexs[home] for ih in range(0, time_step)
        ) + 1e-3, "flex too large"

    def check_share_flex_makes_sense_with_fixed_flex_total(
            self, i_flex, new_batch_flex, loads_next_flex, batch_flex, home, time_step
    ):
        if not self.test:
            return
        assert not (
                0 < i_flex < 4
                and new_batch_flex[home][1][i_flex] + loads_next_flex
                > np.sum([batch_flex[home][ih][0] for ih in range(0, time_step + 1)])
                / (1 - self.share_flexs[home]) * self.share_flexs[home] + 1e-3
        ), "loads_next_flex error"
        assert not (
                loads_next_flex > np.sum([batch_flex[home][ih][0] for ih in range(0, time_step + 1)])
        ), "loads_next_flex too large"

    def loads_test(self):
        for home in self.env.homes:
            for ih in range(self.N):
                assert (
                    self.batch['loads'][home, ih]
                    <= self.batch['flex'][home, ih, 0]
                    + self.batch['flex'][home, ih, -1] + 1e-3
                ), "loads too large"

    def check_batch_flex(self, time_step):
        for ih in range(time_step + 1, time_step + self.N):
            if not all(
                    self.batch['loads'][:, ih]
                    <= self.batch_flex[:, ih, 0] + self.batch_flex[:, ih, -1] + 1e-3
            ):
                with open("batch_error", 'wb') as file:
                    pickle.dump(self.batch, file)
                with open("batch_flex_error", 'wb') as file:
                    pickle.dump(self.batch_flex, file)
            assert all(
                self.batch['loads'][:, ih]
                <= self.batch_flex[:, ih, 0] + self.batch_flex[:, ih, -1] + 1e-3
            ), f"time_step {time_step} ih {ih} " \
               f"loads {self.batch['loads'][:, ih]} " \
               f"batch_flex[home][ih] {self.batch_flex[:, ih]} " \
               f"len(batch_flex[0]) {len(self.batch_flex[0])} " \
               f"self.dloaded {self.dloaded}"

