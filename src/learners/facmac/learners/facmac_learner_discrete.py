# adapted from
# https://github.com/oxwhirl/facmac

import copy

import numpy as np
import torch as th
from torch.optim import Adam, RMSprop

from src.learners.facmac.components.episode_buffer import EpisodeBatch
from src.learners.facmac.learners.learner import Learner
from src.learners.facmac.modules.critics.facmac_critic_discrete import \
    FACMACDiscreteCritic
from src.learners.facmac.modules.mixers.qmix import QMixer
from src.learners.facmac.modules.mixers.qmix_ablations import (
    QMixerNonmonotonic, VDNState)
from src.learners.facmac.modules.mixers.vdn import VDNMixer
from src.learners.facmac.utils.rl_utils import build_td_lambda_targets


class FACMACDiscreteLearner(Learner):
    def __init__(self, mac, scheme, rl, N):
        self.__name__ = 'FACMACDiscreteLearner'
        super().__init__(mac, rl, scheme)
        self.rl = rl

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic = FACMACDiscreteCritic(scheme, rl, N)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())
        self.mixer = None
        if rl['mixer'] is not None and self.n_agents > 1:
            # if just 1 agent do not mix anything
            if rl['mixer'] == "vdn":
                self.mixer = VDNMixer()
            elif rl['mixer'] == "qmix":
                self.mixer = QMixer(rl)
            elif rl['mixer'] == "vdn-s":
                self.mixer = VDNState(rl)
            elif rl['mixer'] == "qmix-nonmonotonic":
                self.mixer = QMixerNonmonotonic(rl)
            else:
                raise ValueError(f"Mixer {rl['mixer']} not recognised.")
            self.critic_params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if rl["optimizer"] == "rmsprop":
            self.agent_optimiser = RMSprop(
                params=self.agent_params, lr=rl['facmac']["lr"],
                alpha=rl["optim_alpha"], eps=rl["optim_eps"])
        elif rl["optimizer"] == "adam":
            self.agent_optimiser = Adam(
                params=self.agent_params, lr=rl['facmac']["lr"],
                eps=rl["optimizer_epsilon"])
        else:
            raise Exception(f"unknown optimizer {rl['optimizer']}")

        if rl["optimizer"] == "rmsprop":
            self.critic_optimiser = RMSprop(
                params=self.critic_params, lr=rl['facmac']["critic_lr"],
                alpha=rl["optim_alpha"], eps=rl["optim_eps"])
        elif rl["optimizer"] == "adam":
            self.critic_optimiser = Adam(
                params=self.critic_params, lr=rl["facmac"]["critic_lr"],
                eps=rl["optimizer_epsilon"])
        else:
            raise Exception(f"unknown optimizer {rl['optimizer']}")

        self.last_target_update_episode = 0
        self.critic_training_steps = 0

    def _compute_critic_and_mixer_q_values_for_batch_states_actions(
            self, batch, actions, critic, mixer, obs, state
    ):
        q_taken, _ = critic(obs, actions)
        if self.n_agents > 1 and self.mixer is not None:
            if self.rl['mixer'] == "vdn":
                q_taken = mixer(
                    q_taken.view(-1, self.n_agents, 1),
                    state
                )
            else:
                q_taken = mixer(
                    q_taken.view(batch.batch_size, -1, 1),
                    state
                )

        return q_taken

    def train(self, batch: EpisodeBatch, t_env: int):
        # Get the relevant quantities
        actions = batch.data.transition_data['actions']
        actions = th.tensor(np.around(np.array(actions) * self.rl['n_discrete_actions']))
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # Train the critic batched
        target_mac_out = self._get_target_actions_batch(batch, t_env)

        q_taken = self._compute_critic_and_mixer_q_values_for_batch_states_actions(
            batch, actions[:, :-1], self.critic, self.mixer,
            batch["obs"][:, :-1], batch["state"][:, :-1]
        )
        target_vals = self._compute_critic_and_mixer_q_values_for_batch_states_actions(
            batch, target_mac_out.detach(), self.target_critic, self.target_mixer,
            batch["obs"][:, :], batch["state"][:, :]
        )

        if self.n_agents > 1 and self.mixer is not None:
            q_taken = q_taken.view(batch.batch_size, -1, 1)
            target_vals = target_vals.view(batch.batch_size, -1, 1)
        else:
            q_taken = q_taken.view(batch.batch_size, -1, self.n_agents)
            target_vals = target_vals.view(batch.batch_size, -1, self.n_agents)

        targets = build_td_lambda_targets(
            batch["reward"], terminated, mask, target_vals,
            self.n_agents, self.rl["facmac"]["gamma"], self.rl["td_lambda"])
        mask = mask[:, :-1]

        self.compute_grad_loss(q_taken, targets, mask)
        th.nn.utils.clip_grad_norm_(
            self.critic_params, self.rl["grad_norm_clip"]
        )
        self.critic_optimiser.step()
        self.critic_training_steps += 1

        # Train the actor
        # Use gumbel softmax to reparameterize the stochastic policies
        # as deterministic functions of independent
        # noise to compute the policy gradient
        # (one hot action input to the critic)
        chosen_action_qvals = self._compute_critic_chosen_actions_qvals(batch, t_env)

        if self.n_agents > 1 and self.mixer is not None:
            if self.rl['mixer'] == "vdn":
                chosen_action_qvals = self.mixer(
                    chosen_action_qvals.view(-1, self.n_agents, 1),
                    batch["state"][:, :-1])
                chosen_action_qvals = chosen_action_qvals.view(
                    batch.batch_size, -1, 1)
            else:
                chosen_action_qvals = self.mixer(
                    chosen_action_qvals.view(batch.batch_size, -1, 1),
                    batch["state"][:, :-1])

        # Compute the actor loss
        pg_loss = - (chosen_action_qvals * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        self.agent_optimiser.step()
        if self.rl["target_update_mode"] == "hard":
            if (self.critic_training_steps - self.last_target_update_episode) \
                    / self.rl["target_update_interval"] >= 1.0:
                self._update_targets()
                self.last_target_update_episode = self.critic_training_steps
        elif self.rl["target_update_mode"] \
                in ["soft", "exponential_moving_average"]:
            self._update_targets_soft(
                tau=self.rl["target_update_tau"]
            )
        else:
            raise Exception(f"unknown target update mode {self.rl['target_update_tau']}")

    def _compute_critic_chosen_actions_qvals(self, batch, t_env):
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            act_outs = self.mac.select_actions(
                batch, t_ep=t, t_env=t_env, test_mode=False, explore=False)
            mac_out.append(act_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        chosen_action_qvals, _ = self.critic(batch["obs"][:, :-1], mac_out)

        return chosen_action_qvals
