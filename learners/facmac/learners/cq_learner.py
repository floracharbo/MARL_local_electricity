# adapted from
# https://github.com/oxwhirl/facmac

import copy

import torch as th
from torch.optim import Adam, RMSprop

from learners.facmac.components.episode_buffer import EpisodeBatch
from learners.facmac.learners.learner import Learner


class CQLearner(Learner):
    def __init__(self, mac, scheme, rl):
        self.__name__ = 'CQLearner'
        super().__init__(mac, rl, scheme)

        self.last_target_update_episode = 0

        if rl['mixer'] is not None and rl['n_agents'] > 1:
            self.agent_params += list(self.mixer.parameters())
            self.named_params.update(dict(self.mixer.named_parameters()))
            self.target_mixer = copy.deepcopy(self.mixer)

        if rl["optimizer"] == "rmsprop":
            self.optimiser = RMSprop(params=self.agent_params,
                                     alpha=rl['optim_alpha'],
                                     lr=rl['lr'],
                                     eps=rl['optim_eps'])
        elif rl["optimizer"] == "adam":
            self.optimiser = Adam(params=self.agent_params,
                                  lr=rl['lr'],
                                  eps=rl["optimizer_epsilon"])
        else:
            raise Exception("unknown optimizer {}".format(rl["optimizer"]))

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        chosen_action_qvals = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            # Note the minimum value of max_seq_length is 2
            agent_outs, _ = self.mac.forward(
                batch, actions=batch["actions"][:, t:t + 1].detach(), t=t)
            chosen_action_qvals.append(agent_outs)

        # Concat over time
        chosen_action_qvals = th.stack(chosen_action_qvals[:-1], dim=1)

        best_target_actions = self._get_target_actions_batch(batch, t_env=None)

        target_max_qvals = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _ = self.target_mac.forward(
                batch, t=t, actions=best_target_actions[:, t].detach())
            target_max_qvals.append(target_agent_outs)

        # Concat over time
        target_max_qvals = th.stack(target_max_qvals[1:], dim=1)

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(
                chosen_action_qvals.view(-1, self.rl['n_agents'], 1),
                batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(
                target_max_qvals.view(-1, self.rl['n_agents'], 1),
                batch["state"][:, 1:])
            chosen_action_qvals = chosen_action_qvals.view(
                batch.batch_size, -1, 1)
            target_max_qvals = target_max_qvals.view(batch.batch_size, -1, 1)
        else:
            chosen_action_qvals = chosen_action_qvals.view(
                batch.batch_size, -1, self.rl['n_agents'])
            target_max_qvals = target_max_qvals.view(
                batch.batch_size, -1, self.rl['n_agents'])

        # Calculate 1-step Q-Learning targets
        targets = rewards.expand_as(target_max_qvals) \
            + self.rl['facmac']['gamma'] \
            * (1 - terminated.expand_as(target_max_qvals)) \
            * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        assert self.rl['runner_scope'] == "episodic",\
            "Runner scope HAS to be episodic if using rnn!"
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        if self.rl["target_update_mode"] == 'hard':
            if (episode_num - self.last_target_update_episode) \
                    / self.rl['target_update_interval'] >= 1.0:
                self._update_targets()
                self.last_target_update_episode = episode_num
        elif self.rl["target_update_mode"] in \
                ["soft", "exponential_moving_average"]:
            self._update_targets_soft(tau=self.rl["target_update_tau"])
        else:
            target_update_mode = self.rl["target_update_mode"], "hard"
            raise Exception(f"unknown target update mode: "
                            f"{target_update_mode}!")

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")
