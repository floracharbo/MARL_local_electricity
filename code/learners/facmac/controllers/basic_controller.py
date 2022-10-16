# adapted from
# https://github.com/oxwhirl/facmac

import torch as th
from learners.facmac.modules.agents import REGISTRY as agent_REGISTRY


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, rl):
        self.n_agents = rl['n_homes']
        self.rl = rl
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = rl['agent_output_type']

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None),
                       test_mode=False, explore=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(
            ep_batch, t_ep, return_logits=(not test_mode))
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env,
            test_mode=test_mode, explore=explore)

        return chosen_actions

    def forward(self, ep_batch, t_ep):
        agent_inputs = self._build_inputs(ep_batch, t_ep)
        agent_outs, self.hidden_states = self.agent(
            agent_inputs, self.hidden_states)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(
            batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def named_parameters(self):
        return self.agent.named_parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def load_state_from_state_dict(self, state_dict):
        self.agent.load_state_dict(state_dict)

    def cuda(self, device="cuda"):
        self.agent.cuda(device=device)

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.rl['agent_facmac']](
            input_shape, self.rl)

    def share(self):
        self.agent.share_memory()

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.rl['obs_last_action']:

            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])

        if self.rl['obs_agent_id']:
            inputs.append(th.eye(
                self.n_agents, device=batch.device).unsqueeze(0).expand(
                bs, -1, -1))

        try:
            inputs = th.cat([x.reshape(bs * self.n_agents, -1)
                             for x in inputs], dim=1)
        except Exception:
            pass
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.rl['obs_last_action']:
            input_shape += scheme["actions_onehot"]["vshape"][0]

        if self.rl['obs_agent_id']:
            input_shape += self.n_agents

        return input_shape

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load("{}/agent.th".format(path),
                    map_location=lambda storage, loc: storage))
