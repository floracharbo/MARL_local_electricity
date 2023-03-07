# adapted from
# https://github.com/oxwhirl/facmac

import copy

import numpy as np
import torch as th
from torch.optim import Adam, RMSprop

from src.learners.facmac.components.episode_buffer import EpisodeBatch
from src.learners.facmac.learners.learner import Learner
from src.learners.facmac.modules.critics.facmac_critic import FACMACCritic


class FACMACLearner(Learner):
    def __init__(self, mac, scheme, rl, N):
        self.__name__ = 'FACMACLearner'
        super().__init__(mac, rl, scheme)

        self.critic = FACMACCritic(scheme, rl, N)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())
        if rl['mixer'] is not None and self.n_agents > 1:
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
            raise Exception(f"unknown optimizer {rl['optimizer']}")

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

    def get_critic_outs(self, critic, actions_, mixer, batch):
        critic.init_hidden(batch.batch_size)
        list_critic_out = []
        for t in range(batch.max_seq_length):
            inputs = self._build_inputs(batch, t=t)
            critic_out, critic.hidden_states = critic(
                inputs, actions_[:, t:t + 1].detach(),
                critic.hidden_states
            )
            if self.n_agents > 1 and self.mixer is not None:
                critic_out = mixer(critic_out.view(
                    batch.batch_size, -1, 1), batch["state"][:, t: t + 1])
            list_critic_out.append(critic_out)
        list_critic_out = th.stack(list_critic_out, dim=1)

        return list_critic_out

    def _train_critic(self, batch):
        rewards = batch.data.transition_data['reward']
        if self.rl['trajectory']:
            rewards = sum(rewards)
        actions = batch.data.transition_data['actions']
        terminated = batch.data.transition_data['terminated']
        mask = batch.data.transition_data['filled']
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        target_actions = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_target_outs = self.target_mac.select_actions(
                batch, t_ep=t, t_env=None, test_mode=True,
                critic=self.target_critic,
            )
            assert not th.isnan(agent_target_outs[0][0][0]), \
                "agent_target_outs nan"
            target_actions.append(agent_target_outs)
        target_actions = th.stack(target_actions, dim=1)  # Concat over time

        # replace all nan actions with the target action
        # so the gradient will be zero
        if self.rl['no_flex_action'] == 'target':
            actions, indexes_nan = self._replace_nan_actions_with_target_actions(
                actions, target_actions
            )

        list_critics = [self.critic, self.target_critic]
        list_actions = [actions, target_actions]
        if self.mixer is not None and self.n_agents > 1:
            list_mixers = [self.mixer, self.target_mixer]
        else:
            list_mixers = [None, None]
        q_taken, target_vals = [
            self.get_critic_outs(critic, actions_, mixer, batch)
            for critic, actions_, mixer in zip(list_critics, list_actions, list_mixers)
        ]

        if self.n_agents > 1 and self.mixer is not None:
            q_taken = q_taken.view(batch.batch_size, -1, 1)
            target_vals = target_vals.view(batch.batch_size, -1, 1)
        else:
            q_taken = q_taken.view(batch.batch_size, -1, self.n_agents)
            target_vals = target_vals.view(batch.batch_size, -1, self.n_agents)

        target_vals, terminated, rewards, q_taken = self._vars_to_cuda(
            target_vals, terminated, rewards, q_taken
        )

        targets = rewards.expand_as(target_vals) \
            + self.rl['facmac']['gamma'] * (1 - terminated.expand_as(target_vals)) * target_vals
        td_error = (targets.detach() - q_taken)
        td_error = self.add_supervised_loss(td_error, batch)

        mask = mask.expand_as(td_error)
        mask = mask.cuda() if self.cuda_available else mask
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()
        lr = self.critic_lr if (not self.hysteretic or loss > 0) \
            else self.critic_lr * self.beta_to_alpha
        for g in self.critic_optimiser.param_groups:
            g['lr'] = lr

        self.critic_optimiser.zero_grad()
        loss.backward()
        self.critic_optimiser.step()

    def _train_actor(self, batch):
        # Optimize over the entire joint action space
        mac_out = []
        chosen_action_qvals = []
        self.mac.init_hidden(batch.batch_size)
        self.critic.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(
                batch, t=t, select_actions=True
            )["actions"].view(batch.batch_size, self.n_agents, self.n_actions)
            q, self.critic.hidden_states = self.critic(
                self._build_inputs(batch, t=t), agent_outs,
                self.critic.hidden_states)

            if self.n_agents > 1 and self.mixer is not None:
                q = self.mixer(q.view(batch.batch_size, -1, 1),
                               batch["state"][:, t: t + 1])

            mac_out.append(agent_outs)
            chosen_action_qvals.append(q)
        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.stack(chosen_action_qvals, dim=1)
        pi = mac_out

        chosen_action_qvals = self.add_supervised_loss(chosen_action_qvals, batch)
        # Compute the actor loss
        pg_loss = - chosen_action_qvals.mean() + (pi ** 2).mean() * 1e-3
        lr = self.lr if (not self.hysteretic or pg_loss > 0) \
            else self.lr * self.beta_to_alpha
        for g in self.agent_optimiser.param_groups:
            g['lr'] = lr

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

    def train(self, batch: EpisodeBatch, t_env):
        # Train the critic batched
        self._train_critic(batch)

        # Train the actor
        self._train_actor(batch)

    def add_supervised_loss(self, td_error, batch):
        if self.rl['supervised_loss']:
            for episode in range(batch.batch_size):
                for step in range(batch.max_seq_length):
                    if batch.data.transition_data['optimal_actions'][episode, step][0][0] != -1:
                        td_error[episode, step] += th.sum(th.square(
                            batch.data.transition_data['actions'][episode][step]
                            - batch.data.transition_data['optimal_actions'][episode, step]
                        )) * self.rl['supervised_loss_weight']

        return td_error

    def _replace_nan_actions_with_target_actions(self, actions, target_actions):
        shape_actions = np.shape(actions)
        indexes_nan = []

        for i_batch in range(shape_actions[0]):
            for time_step in range(shape_actions[1]):
                for i_agent in range(shape_actions[2]):
                    for i_action in range(shape_actions[3]):
                        if th.isnan(
                                actions[i_batch, time_step, i_agent, i_action]
                        ):
                            indexes_nan.append([i_batch, time_step, i_agent, i_action])
                            actions[i_batch, time_step, i_agent, i_action] \
                                = target_actions[
                                i_batch, time_step, i_agent, i_action]

        # print(f"indexes_nan [i_batch, time_step, i_agent, i_action] {indexes_nan}")

        return actions, indexes_nan

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []

        if self.rl['recurrent_critic']:
            # The individual Q conditions on the global
            # action-observation history and individual action
            inputs.append(batch["obs"][:, t].repeat(
                1, self.n_agents, 1).view(
                bs, self.n_agents, -1))
            if self.rl['obs_last_action']:
                if t == 0:
                    inputs.append(th.zeros_like(
                        batch["actions"][:, t].repeat(
                            1, self.n_agents, 1).
                        view(bs, self.n_agents, -1)))
                else:
                    inputs.append(
                        batch["actions"][:, t - 1].repeat(1, self.n_agents, 1).view(
                            bs, self.n_agents, -1
                        )
                    )

        else:
            inputs.append(batch["obs"][:, t])

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)

        return inputs

    def _vars_to_cuda(self, target_vals, terminated, rewards, q_taken):
        if self.cuda_available:
            target_vals = target_vals.cuda()
            terminated = terminated.cuda()
            rewards = rewards.cuda()
            q_taken = q_taken.cuda()

        return target_vals, terminated, rewards, q_taken
