# adapted from
# https://github.com/oxwhirl/facmac

from src.learners.facmac.modules.critics.critic import Critic

import torch as th
import torch.nn.functional as F


class MADDPGCritic(Critic):
    def __init__(self, scheme, rl):
        super().__init__(rl)

        # The centralized critic takes the state input, not observation
        self.input_shape = \
            scheme["state"]["vshape"] + self.n_actions * self.n_agents

        # Set up network layers
        self.set_up_network_layers(rl)

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
