import copy

import numpy as np
import torch as th
from torch.optim import Adam, RMSprop

from learners.facmac.components.episode_buffer import EpisodeBatch
from learners.facmac.modules.critics.facmac import FACMACCritic
from learners.facmac.modules.mixers.qmix import QMixer
from learners.facmac.modules.mixers.vdn import VDNMixer

# from learners.facmac.modules.mixers.qmix_ablations
# import VDNState, QMixerNonmonotonic


class FACMACLearner:
    def __init__(self, mac, scheme, rl):
        self.rl = rl
        self.n_agents = rl['n_agents']
        self.n_actions = rl['dim_actions']
        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())
        self.cuda_available = True if th.cuda.is_available() else False

        self.critic = FACMACCritic(scheme, rl)

        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        self.mixer = None
        if rl['mixer'] is not None \
                and self.rl['n_agents'] > 1:
            # if just 1 agent do not mix anything
            if rl['mixer'] == "vdn":
                self.mixer = VDNMixer()
            elif rl['mixer'] == "qmix":
                self.mixer = QMixer(rl)
            # elif rl['mixer'] == "vdn-s":
            #     self.mixer = VDNState(rl)
            # elif rl['mixer'] == "qmix-nonmonotonic":
            #     self.mixer = QMixerNonmonotonic(rl)
            else:
                raise ValueError(f"Mixer {rl['mixer']} not recognised.")
            self.critic_params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if rl["optimizer"] == "rmsprop":
            self.agent_optimiser = RMSprop(
                params=self.agent_params, lr=rl['lr'],
                alpha=rl['optim_alpha'], eps=rl['optim_eps'])
        elif rl["optimizer"] == "adam":
            self.agent_optimiser = Adam(
                params=self.agent_params, lr=rl['lr'],
                eps=rl['optimizer_epsilon'])
        else:
            raise Exception("unknown optimizer {}".format(rl["optimizer"]))

        if rl["optimizer"] == "rmsprop":
            self.critic_optimiser = RMSprop(
                params=self.critic_params, lr=rl['facmac']['critic_lr'],
                alpha=rl['optim_alpha'], eps=rl['optim_eps'])
        elif rl["optimizer"] == "adam":
            self.critic_optimiser = Adam(
                params=self.critic_params, lr=rl['facmac']['critic_lr'],
                eps=rl['optimizer_epsilon'])
        else:
            raise Exception("unknown optimizer {}".format(rl["optimizer"]))

        # self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
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
        self.critic.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            inputs = self._build_inputs(batch, t=t)
            critic_out, self.critic.hidden_states = self.critic(
                inputs, actions[:, t:t + 1].detach(),
                self.critic.hidden_states)
            if self.mixer is not None:
                critic_out = self.mixer(critic_out.view(
                    batch.batch_size, -1, 1), batch["state"][:, t: t + 1])
            q_taken.append(critic_out)
        q_taken = th.stack(q_taken, dim=1)

        target_vals = []
        self.target_critic.init_hidden(batch.batch_size)
        for t in range(1, batch.max_seq_length):
            target_inputs = self._build_inputs(batch, t=t)
            target_critic_out, self.target_critic.hidden_states = \
                self.target_critic(target_inputs,
                                   target_actions[:, t: t + 1].detach(),
                                   self.target_critic.hidden_states)
            if self.mixer is not None:
                target_critic_out = self.target_mixer(
                    target_critic_out.view(batch.batch_size, -1, 1),
                    batch["state"][:, t: t + 1])
            target_vals.append(target_critic_out)
        target_vals = th.stack(target_vals, dim=1)

        if self.mixer is not None:
            q_taken = q_taken.view(batch.batch_size, -1, 1)
            target_vals = target_vals.view(batch.batch_size, -1, 1)
        else:
            q_taken = q_taken.view(batch.batch_size, -1, self.n_agents)
            target_vals = target_vals.view(batch.batch_size, -1, self.n_agents)

        target_vals = target_vals.cuda() if self.cuda_available else target_vals
        terminated = terminated.cuda() if self.cuda_available else terminated
        rewards = rewards.cuda() if self.cuda_available else rewards
        q_taken = q_taken.cuda() if self.cuda_available else q_taken

        targets = rewards.expand_as(target_vals) \
            + self.rl['facmac']['gamma'] * \
            (1 - terminated.expand_as(target_vals)) * target_vals
        td_error = (targets.detach() - q_taken)
        mask = mask.expand_as(td_error)
        mask = mask.cuda() if self.cuda_available else mask
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        self.critic_optimiser.step()

        # Train the actor
        # Optimize over the entire joint action space
        mac_out = []
        chosen_action_qvals = []
        self.mac.init_hidden(batch.batch_size)
        self.critic.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(
                batch, t=t, select_actions=True)["actions"].\
                view(batch.batch_size, self.n_agents, self.n_actions)
            q, self.critic.hidden_states = self.critic(
                self._build_inputs(batch, t=t), agent_outs,
                self.critic.hidden_states)
            if self.mixer is not None:
                q = self.mixer(q.view(batch.batch_size, -1, 1),
                               batch["state"][:, t: t + 1])
            mac_out.append(agent_outs)
            chosen_action_qvals.append(q)
        mac_out = th.stack(mac_out[:-1], dim=1)
        chosen_action_qvals = th.stack(chosen_action_qvals[:-1], dim=1)
        pi = mac_out

        # Compute the actor loss
        pg_loss = -chosen_action_qvals.mean() + (pi**2).mean() * 1e-3

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        self.agent_optimiser.step()

        if self.rl['target_update_mode'] == "hard":
            print("hard target update")
            self._update_targets()
        elif self.rl['target_update_mode'] in \
                ["soft", "exponential_moving_average"]:
            self._update_targets_soft(tau=self.rl['target_update_tau'])
        else:
            raise Exception(f"unknown target update mode: "
                            f"{self.rl['target_update_mode']}!")

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(),
                                       self.mac.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau)

        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(),
                                           self.mixer.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []

        if self.rl['recurrent_critic']:
            # The individual Q conditions on the global
            # action-observation history and individual action
            inputs.append(batch["obs"][:, t].repeat(
                1, self.rl['n_agents'], 1).view(
                bs, self.rl['n_agents'], -1))
            if self.rl['obs_last_action']:
                if t == 0:
                    inputs.append(th.zeros_like(
                        batch["actions"][:, t].repeat(
                            1, self.rl['n_agents'], 1).
                        view(bs, self.rl['n_agents'], -1)))
                else:
                    inputs.append(batch["actions"][:, t - 1].repeat(
                        1, self.rl['n_agents'], 1).view(
                        bs, self.rl['n_agents'], -1))

        else:
            inputs.append(batch["obs"][:, t])

        inputs = th.cat([x.reshape(bs * self.n_agents, -1)
                         for x in inputs], dim=1)
        return inputs

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        # self.logger.console_logger.info("Updated all target networks")

    def cuda(self, device="cuda:0"):
        self.mac.cuda(device=device)
        self.target_mac.cuda(device=device)
        self.critic.cuda(device=device)
        self.target_critic.cuda(device=device)
        if self.mixer is not None:
            self.mixer.cuda(device=device)
            self.target_mixer.cuda(device=device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load("{}/mixer.th".format(path),
                        map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(
            th.load("{}/opt.th".format(path),
                    map_location=lambda storage, loc: storage))
