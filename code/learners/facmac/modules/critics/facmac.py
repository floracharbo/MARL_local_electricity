# adapted from
# https://github.com/oxwhirl/facmac

from code.learners.facmac.modules.critics.critic import Critic

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FACMACCritic(Critic):
    def __init__(self, scheme, rl):
        super().__init__(rl)

        self.input_shape = scheme["obs"]["vshape"] + self.n_actions
        self.hidden_states = None

        # Set up network layers
        self.set_up_network_layers(rl)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = inputs.cuda() if self.cuda_available else inputs
            actions = actions.cuda() if self.cuda_available else actions
            inputs = th.cat(
                [inputs.view(-1, self.input_shape - self.n_actions),
                 actions.contiguous().view(-1, self.n_actions)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)

        return q, hidden_state


class FACMACDiscreteCritic(nn.Module):
    def __init__(self, scheme, rl):
        super().__init__(rl)

        self.input_shape = scheme["obs"]["vshape"] + self.n_actions
        self.hidden_states = None

        # Set up network layers
        self.set_up_network_layers(rl)

    def init_hidden(self):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat(
                [inputs.reshape(-1, self.input_shape - self.n_actions),
                 actions.contiguous().view(-1, self.n_actions)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)

        return q, hidden_state
