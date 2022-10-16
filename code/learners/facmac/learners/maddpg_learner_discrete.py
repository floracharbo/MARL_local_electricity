# adapted from
# https://github.com/oxwhirl/facmac

import torch as th
from learners.facmac.components.episode_buffer import EpisodeBatch
from learners.facmac.learners.learner import Learner
from learners.facmac.utils.rl_utils import build_td_lambda_targets


class MADDPGDiscreteLearner(Learner):
    def __init__(self, mac, scheme, rl):
        self.__name__ = 'MADDPGDiscreteLearner'
        super().__init__(mac, rl, scheme)
        self.last_target_update_episode = 0
        self.critic_training_steps = 0

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        actions = batch["actions_onehot"][:, :]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # Train the critic batched
        target_mac_out = self._get_target_actions_batch(batch, t_env)

        q_taken, _ = self.critic(batch["state"][:, :-1], actions[:, :-1])
        target_vals, _ = self.target_critic(
            batch["state"][:, :], target_mac_out.detach())

        q_taken = q_taken.view(batch.batch_size, -1, 1)
        target_vals = target_vals.view(batch.batch_size, -1, 1)
        targets = build_td_lambda_targets(
            batch["reward"], terminated, mask, target_vals,
            self.n_agents, self.args.gamma, self.args.td_lambda)
        mask = mask[:, :-1]

        masked_td_error, loss = self.compute_grad_loss(q_taken, targets, mask)

        critic_grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()
        self.critic_training_steps += 1

        chosen_action_qvals = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.select_actions(
                batch, t_ep=t, t_env=t_env, test_mode=False, explore=False)

            chosen_action_qvals = self._append_chosen_action_qvals(
                actions, batch, agent_outs, chosen_action_qvals, t
            )

        chosen_action_qvals = th.stack(chosen_action_qvals, dim=1)

        # Compute the actor loss
        pg_loss = - chosen_action_qvals.mean()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(
            self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        target_update_mode = \
            getattr(self.args, "target_update_mode", "hard")
        if target_update_mode == "hard":
            if (self.critic_training_steps - self.last_target_update_episode) \
                    / self.args.target_update_interval >= 1.0:
                self._update_targets()
                self.last_target_update_episode = self.critic_training_steps
        elif target_update_mode in ["soft", "exponential_moving_average"]:
            self._update_targets_soft(
                tau=getattr(self.args, "target_update_tau", 0.001))
        else:
            raise Exception(
                "unknown target update mode: {target_update_mode}!")

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs",
                masked_td_error.abs().sum().item() / mask_elems,
                t_env)
            self.logger.log_stat(
                "q_taken_mean",
                (q_taken * mask).sum().item() / mask_elems,
                t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", agent_grad_norm, t_env)
            self.log_stats_t = t_env

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []

        # The centralized critic takes the state input, not observation
        inputs.append(batch["state"][:, t])

        if self.args.recurrent_critic:
            if self.args.obs_last_action:
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions"][:, t]))
                else:
                    inputs.append(batch["actions"][:, t - 1])

        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=1)
        return inputs
