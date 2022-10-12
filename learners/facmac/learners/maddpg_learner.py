# adapted from
# https://github.com/oxwhirl/facmac

import copy

import torch as th
from torch.optim import Adam, RMSprop

from learners.facmac.components.episode_buffer import EpisodeBatch
from learners.facmac.learners.learner import Learner
from learners.facmac.modules.critics.maddpg import MADDPGCritic
from learners.facmac.utils.rl_utils import input_last_action

class MADDPGLearner(Learner):
    def __init__(self, mac, scheme, rl):
        self.rl = rl
        self.n_agents = rl['n_agents']
        self.n_actions = rl['dim_actions']

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic = MADDPGCritic(scheme, rl)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        if self.rl['optimizer'] == "rmsprop":
            self.agent_optimiser = RMSprop(params=self.agent_params,
                                           lr=rl['lr'],
                                           alpha=self.rl['optim_alpha'],
                                           eps=self.rl['optim_eps'])
        elif self.rl['optimizer'] == "adam":
            self.agent_optimiser = Adam(params=self.agent_params,
                                        lr=rl['lr'],
                                        eps=self.rl['optimizer_epsilon'])
        else:
            raise Exception(f"unknown optimizer {self.rl['optimizer']}")

        if self.rl['optimizer'] == "rmsprop":
            self.critic_optimiser = RMSprop(params=self.critic_params,
                                            lr=self.rl['facmac']['critic_lr'],
                                            alpha=self.rl['optim_alpha'],
                                            eps=self.rl['optim_eps'])
        elif self.rl['optimizer'] == "adam":
            self.critic_optimiser = Adam(params=self.critic_params,
                                         lr=self.rl['facmac']['critic_lr'],
                                         eps=self.rl['optimizer_epsilon'])
        else:
            raise Exception(f"unknown optimizer {self.rl['optimizer']}")

        # self.log_stats_t = -self.args.learner_log_interval - 1

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

        td_error = (q_taken - targets.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        self.critic_optimiser.step()

        mac_out = []
        chosen_action_qvals = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(
                batch, t=t, select_actions=True)["actions"].view(
                batch.batch_size, self.n_agents, self.n_actions)

            chosen_action_qvals = self._append_chosen_action_qvals(
                actions, batch, agent_outs, chosen_action_qvals, t
            )

            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.stack(chosen_action_qvals, dim=1)
        pi = mac_out

        # Compute the actor loss
        pg_loss = -chosen_action_qvals.mean() + (pi**2).mean() * 1e-3

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

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated all target networks")

    def cuda(self, device="cuda:0"):
        self.mac.cuda(device=device)
        self.target_mac.cuda(device=device)
        self.critic.cuda(device=device)
        self.target_critic.cuda(device=device)

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.agent_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.agent_optimiser.load_state_dict(
            th.load("{}/opt.th".format(path),
                    map_location=lambda storage, loc: storage))
