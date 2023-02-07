# adapted from
# https://github.com/oxwhirl/facmac


import torch as th

from src.learners.facmac.components.episode_buffer import EpisodeBatch
from src.learners.facmac.learners.learner import Learner
from src.learners.facmac.utils.rl_utils import input_last_action


class MADDPGLearner(Learner):
    def __init__(self, mac, scheme, rl):
        self.__name__ = 'MADDPGLearner'
        super().__init__(mac, scheme, rl)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # Train the critic batched
        target_actions = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_target_outs = self.target_mac.select_actions(
                batch, t_ep=t, t_env=None, test_mode=True,
                critic=self.target_critic, target_mac=True)
            target_actions.append(agent_target_outs)
        target_actions = th.stack(target_actions, dim=1)  # Concat over time

        q_taken = []
        for t in range(batch.max_seq_length - 1):
            inputs = self._build_inputs(batch, t=t)
            critic_out, _ = self.critic(inputs, actions[:, t: t + 1].detach())
            critic_out = critic_out.view(batch.batch_size, -1, 1)
            q_taken.append(critic_out)
        q_taken = th.stack(q_taken, dim=1)

        target_vals = []
        for t in range(1, batch.max_seq_length):
            target_inputs = self._build_inputs(batch, t=t)
            target_critic_out, _ = self.target_critic(
                target_inputs, target_actions[:, t: t + 1].detach())
            target_critic_out = target_critic_out.view(batch.batch_size, -1, 1)
            target_vals.append(target_critic_out)
        target_vals = th.stack(target_vals, dim=1)

        q_taken = q_taken.view(batch.batch_size, -1, 1)
        target_vals = target_vals.view(batch.batch_size, -1, 1)
        targets = rewards.expand_as(target_vals) \
            + self.rl['facmac']['gamma'] \
            * (1 - terminated.expand_as(target_vals)) * target_vals

        self.compute_grad_loss(q_taken, targets, mask)

        self.critic_optimiser.step()

        mac_out = []
        chosen_action_qvals = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(
                batch, t=t, select_actions=True).view(
                batch.batch_size, self.n_agents, self.n_actions)

            chosen_action_qvals = self._append_chosen_action_qvals(
                actions, batch, agent_outs, chosen_action_qvals, t
            )

            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.stack(chosen_action_qvals, dim=1)
        pi = mac_out

        # Compute the actor loss
        pg_loss = - chosen_action_qvals.mean() + (pi**2).mean() * 1e-3

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        self.agent_optimiser.step()

        if self.rl['target_update_mode'] == "hard":
            self._update_targets()
        elif self.rl['target_update_mode'] in \
                ["soft", "exponential_moving_average"]:
            self._update_targets_soft(tau=self.rl["target_update_tau"])
        else:
            raise Exception(
                f"unknown target update mode: {self.rl['target_update_mode']}")

    def _build_inputs(self, batch, t):
        bs = batch.batch_size

        # The centralized critic takes the state input, not observation
        inputs = [batch["state"][:, t]]

        if self.rl['recurrent_critic']:
            inputs = input_last_action(
                self.rl['obs_last_action'], inputs, batch, t
            )

        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=1)
        return inputs
