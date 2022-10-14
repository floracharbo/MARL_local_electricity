# adapted from
# https://github.com/oxwhirl/facmac

import copy

import numpy as np
import torch as th
from torch.optim import Adam, RMSprop

from learners.facmac.components.episode_buffer import EpisodeBatch
from learners.facmac.learners.learner import Learner
from learners.facmac.modules.critics.facmac import FACMACCritic

# from learners.facmac.modules.mixers.qmix_ablations
# import VDNState, QMixerNonmonotonic


class FACMACLearner(Learner):
    def __init__(self, mac, scheme, rl):
        self.__name__ = 'FACMACLearner'
        super.__init__(mac, rl, scheme)

        self.critic = FACMACCritic(scheme, rl)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        if rl['mixer'] is not None \
                and self.rl['n_agents'] > 1:
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
                critic=self.target_critic)
            assert not th.isnan(agent_target_outs[0][0][0]), \
                "agent_target_outs nan"
            target_actions.append(agent_target_outs)
        target_actions = th.stack(target_actions, dim=1)  # Concat over time
        q_taken = []

        # replace all nan actions with the target action
        # so the gradient will be zero
        shape_actions = np.shape(actions)
        for i_batch in range(shape_actions[0]):
            for i_step in range(shape_actions[1]):
                for i_agent in range(shape_actions[2]):
                    for i_action in range(shape_actions[3]):
                        if th.isnan(
                                actions[i_batch, i_step, i_agent, i_action]
                        ):

                            actions[i_batch, i_step, i_agent, i_action] \
                                = target_actions[
                                i_batch, i_step, i_agent, i_action]
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

        if self.cuda_available:
            target_vals = target_vals.cuda()
            terminated = terminated.cuda()
            rewards = rewards.cuda()
            q_taken = q_taken.cuda()

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
        pg_loss = - chosen_action_qvals.mean() + (pi**2).mean() * 1e-3
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
