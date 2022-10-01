# adapted from
# https://github.com/oxwhirl/facmac

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MADDPGCritic(nn.Module):
    def __init__(self, scheme, rl):
        super(MADDPGCritic, self).__init__()
        self.rl = rl
        self.n_actions = rl['dim_actions']
        self.n_agents = rl['n_agents']
        self.input_shape = \
            self._get_input_shape(scheme) + self.n_actions * self.n_agents
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, rl['rnn_hidden_dim'])
        self.fc2 = nn.Linear(rl['rnn_hidden_dim'], rl['rnn_hidden_dim'])
        self.fc3 = nn.Linear(rl['rnn_hidden_dim'], 1)

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.reshape(
                -1, self.input_shape - self.n_actions * self.n_agents),
                actions.contiguous().view(-1, self.n_actions * self.n_agents)],
                dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        # The centralized critic takes the state input, not observation
        input_shape = scheme["state"]["vshape"]
        return input_shape
