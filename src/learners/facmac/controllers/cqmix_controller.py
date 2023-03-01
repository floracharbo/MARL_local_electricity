# adapted from
# https://github.com/oxwhirl/facmac

import numpy as np
import torch as th
import torch.distributions as tdist
from gym import spaces

from src.learners.facmac.utils.rl_utils import input_last_action

from .basic_controller import BasicMAC


# This multi-agent controller shares parameters between agents
class CQMixMAC(BasicMAC):

    def _exploration_noising(
            self, test_mode, chosen_actions, t_env,
            explore_agent_ids, action_space, ep_batch, bs
    ):
        if not test_mode:  # do exploration
            if self.rl["exploration_mode"] == "ornstein_uhlenbeck":
                x = getattr(self, "ou_noise_state", chosen_actions.clone().zero_())
                mu = 0
                theta = self.rl['ou_theta']
                sigma = self.rl['ou_sigma']

                noise_scale = self.rl['ou_noise_scale'] \
                    if t_env < self.rl["episode_limit"] \
                    * self.rl["ou_stop_episode"] \
                    else 0.0

                dx = theta * (mu - x) + sigma * x.clone().normal_()
                self.ou_noise_state = x + dx
                ou_noise = self.ou_noise_state * noise_scale
                chosen_actions = chosen_actions + ou_noise
            elif self.rl["exploration_mode"] == "gaussian":
                start_steps = self.rl["start_steps"]
                act_noise = self.rl["act_noise"]

                if t_env >= start_steps:
                    if explore_agent_ids is None:
                        x = chosen_actions.clone().zero_()
                        chosen_actions += act_noise * x.clone().normal_()
                    else:
                        for idx in explore_agent_ids:
                            x = chosen_actions[:, idx].clone().zero_()
                            chosen_actions[:, idx] \
                                += act_noise * x.clone().normal_()
                else:
                    chosen_actions = th.from_numpy(
                        np.array([[action_space[i].sample()
                                   for i in range(self.n_homes)]
                                  for _ in range(ep_batch[bs].batch_size)])
                    ).float().to(device=ep_batch.device)

        return chosen_actions

    def _clamp_actions(self, action_space, chosen_actions):
        if all([isinstance(act_space, spaces.Box)
                for act_space in action_space]):
            for _aid in range(self.n_homes):
                for _actid in range(action_space[_aid].shape[0]):
                    chosen_actions[:, _aid, _actid].clamp_(
                        (action_space[_aid].low[_actid]).item(),
                        (action_space[_aid].high[_actid]).item())
        elif all([isinstance(act_space, spaces.Tuple)
                  for act_space in action_space]):
            # NOTE: This was added to handle scenarios
            # like simple_reference since action space is Tuple
            for _aid in range(self.n_homes):
                n = action_space[_aid].spaces[0].shape[0]
                for _actid in range(n):
                    chosen_actions[:, _aid, _actid].clamp_(
                        action_space[_aid].spaces[0].low[_actid],
                        action_space[_aid].spaces[0].high[_actid])

                n = action_space[_aid].spaces[1].shape[0]
                for _actid in range(n):
                    tmp_idx = \
                        _actid\
                        + action_space[_aid].spaces[0].shape[0]
                    chosen_actions[:, _aid, tmp_idx].clamp_(
                        action_space[_aid].spaces[1].low[_actid],
                        action_space[_aid].spaces[1].high[_actid])

        return chosen_actions

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None),
                       test_mode=False, critic=None,
                       explore_agent_ids=None):

        if self.rl['agent_facmac'] == "cqmix":
            raise Exception("No CQMIX agent selected (naf, icnn, qtopt)!")

        action_space = self.rl['action_space']

        rdn_eps = np.random.rand()
        rdn_action = th.rand((ep_batch[bs].batch_size, self.n_homes, self.rl['dim_actions']))
        if not test_mode and self.rl['exploration_mode'] == 'eps_greedy' and rdn_eps < self.rl['facmac']['epsilon']:
            chosen_actions = th.tensor(self.rl['low_actions']) + rdn_action * (1 - self.rl['low_actions'])
        # Note batch_size_run is set to be 1 in our experiments
        elif self.rl['agent_facmac'] in ["naf", "mlp", "rnn"]:
            hidden_states = self.hidden_states_ih[bs] if self.rl['nn_type'] in ['lstm', 'rnn'] \
                else self.hidden_states[bs]
            chosen_actions = self.forward(
                ep_batch[bs], t_ep,
                hidden_states=hidden_states,
                test_mode=test_mode, select_actions=True
            )["actions"]
            # just to make sure detach
            chosen_actions = chosen_actions.view(
                ep_batch[bs].batch_size, self.n_homes,
                self.rl['dim_actions']
            ).detach()
            pass
        elif self.rl['agent_facmac'] == "icnn":
            inputs = self._build_inputs(ep_batch[bs], t_ep)
            chosen_actions = self.agent.bundle_tuned2(observation=inputs)
            chosen_actions = chosen_actions.view(
                ep_batch[bs].batch_size, self.n_homes,
                self.rl['dim_actions']).detach()

        elif self.rl['agent_facmac'] in ["cem", "cemrnn"]:
            chosen_actions = self.cem_sampling(ep_batch, t_ep, bs)
        elif self.rl['agent_facmac'] in ["cemrand"]:
            N = 64
            agent_inputs = self._build_inputs(ep_batch[bs], t_ep)
            hidden_states = self.hidden_states[bs].repeat(N, 1, 1)

            # Randomly sample N actions from a uniform distribution
            ftype = th.FloatTensor \
                if not next(self.agent.parameters()).is_cuda \
                else th.cuda.FloatTensor
            low = ftype(
                ep_batch[bs].batch_size, self.n_homes,
                self.rl['dim_actions']).zero_() \
                + action_space[0].low[0]
            high = ftype(
                ep_batch[bs].batch_size, self.n_homes,
                self.rl['dim_actions']).zero_() \
                + action_space[0].high[0]
            dist = tdist.Uniform(low.view(-1, self.rl['dim_actions']),
                                 high.view(-1, self.rl['dim_actions']))

            actions = dist.sample((N,)).detach()

            # Pick the best sampled action
            out = self.agent(
                agent_inputs.unsqueeze(0).expand(
                    N, *agent_inputs.shape).contiguous().view(
                    -1, agent_inputs.shape[-1]),
                hidden_states if hidden_states is not None
                else self.hidden_states,
                actions=actions.view(-1, actions.shape[-1]))["Q"].view(
                N, -1, 1)
            topk, topk_idxs = th.topk(out, 1, dim=0)
            action_prime = th.mean(
                actions.gather(0, topk_idxs.repeat(
                    1, 1, self.rl['dim_actions']).long()),
                dim=0)
            chosen_actions = action_prime.clone().view(
                ep_batch[bs].batch_size, self.n_homes,
                self.rl['dim_actions']).detach()
            pass
        else:
            raise Exception(
                f"No known agent type selected for cqmix! "
                f"({self.rl['agent_facmac']})")

        # Now do appropriate noising
        # Ornstein-Uhlenbeck:
        chosen_actions = self._exploration_noising(
            test_mode, chosen_actions, t_env,
            explore_agent_ids, action_space, ep_batch, bs
        )

        # For continuous actions, clamp actions to permissible action range
        # (necessary after exploration)
        chosen_actions = self._clamp_actions(action_space, chosen_actions)

        return chosen_actions

    def get_weight_decay_weights(self):
        return self.agent.get_weight_decay_weights()

    def forward(self, ep_batch, t, actions=None,
                hidden_states=None, select_actions=False,
                test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        ret = self.agent(agent_inputs, self.hidden_states, actions=actions)
        if select_actions:
            self.hidden_states = ret["hidden_state"]
            return ret
        agent_outs = ret["Q"]
        self.hidden_states = ret["hidden_state"]

        if self.agent_output_type == "pi_logits":
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                agent_outs = \
                    ((1 - self.action_selector.epsilon) * agent_outs
                     + th.ones_like(agent_outs)
                     * self.action_selector.epsilon / agent_outs.size(-1))

        return agent_outs.view(ep_batch.batch_size, self.n_homes, -1), actions

    def _build_inputs(self, batch, t, target_mac=False,
                      last_target_action=None):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]  # b1av
        inputs = input_last_action(self.rl['obs_last_action'], inputs, batch, t)
        if self.rl['obs_agent_id']:
            inputs.append(th.eye(
                self.n_homes, device=batch.device).unsqueeze(0).expand(
                bs, -1, -1))

        inputs = th.cat(
            [x.reshape(bs * self.n_homes, - 1) for x in inputs],
            dim=1
        )

        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.rl['obs_last_action']:
            if self.rl['discretize_actions']:
                input_shape += scheme["actions_onehot"]["vshape"][0]
            else:
                input_shape += scheme["actions"]["vshape"][0]
        if self.rl['obs_agent_id']:

            input_shape += self.n_homes

        return input_shape

    def cem_sampling(self, ep_batch, t, bs, critic=None):
        # Number of samples from the param distribution
        N = 64
        # Number of best samples we will consider
        Ne = 6

        ftype = th.FloatTensor \
            if not next(self.agent.parameters()).is_cuda \
            else th.cuda.FloatTensor
        mu = ftype(ep_batch[bs].batch_size, self.n_homes,
                   self.rl['dim_actions']).zero_()
        std = ftype(ep_batch[bs].batch_size, self.n_homes,
                    self.rl['dim_actions']).zero_() + 1.0

        its = 0

        maxits = 2
        agent_inputs = self._build_inputs(ep_batch[bs], t)
        hidden_states = self.hidden_states.reshape(
            -1, self.n_homes, self.rl['rnn_hidden_dim'])[bs].repeat(
            N, 1, 1, 1)

        # Use feed-forward critic here,
        # so it takes only the obs input
        critic_inputs = []
        if critic is not None:
            critic_inputs.append(ep_batch[bs]["obs"][:, t])
            critic_inputs = th.cat(
                [x.reshape(ep_batch[bs].batch_size * self.n_homes, -1)
                 for x in critic_inputs], dim=1)

        while its < maxits:
            dist = tdist.Normal(mu.view(-1, self.rl['dim_actions']),
                                std.view(-1, self.rl['dim_actions']))

            actions = dist.sample((N,)).detach()
            actions_prime = th.tanh(actions)

            if critic is None:
                ret = self.agent(
                    agent_inputs.unsqueeze(0).expand(
                        N, *agent_inputs.shape).contiguous().view(
                        -1, agent_inputs.shape[-1]),
                    hidden_states if hidden_states is not None
                    else self.hidden_states,
                    actions=actions_prime.view(-1, actions_prime.shape[-1]))
                out = ret["Q"].view(N, -1, 1)
            else:
                out, _ = critic(
                    critic_inputs.unsqueeze(0).expand(
                        N, *critic_inputs.shape).contiguous().view(
                        -1, critic_inputs.shape[-1]),
                    actions=actions_prime.view(-1, actions_prime.shape[-1]))
                out = out.view(N, -1, 1)

            topk, topk_idxs = th.topk(out, Ne, dim=0)
            mu = th.mean(actions.gather(
                0, topk_idxs.repeat(1, 1, self.rl['dim_actions']).long()),
                dim=0)
            std = th.std(actions.gather(
                0, topk_idxs.repeat(1, 1, self.rl['dim_actions']).long()),
                dim=0)
            its += 1

        topk, topk_idxs = th.topk(out, 1, dim=0)
        action_prime = th.mean(actions_prime.gather(
            0, topk_idxs.repeat(1, 1, self.rl['dim_actions']).long()
        ), dim=0)
        chosen_actions = action_prime.clone().view(
            ep_batch[bs].batch_size, self.n_homes, self.rl['dim_actions']
        ).detach()

        return chosen_actions
