# adapted from
# https://github.com/oxwhirl/facmac

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from src.learners.facmac.modules.critics.critic import Critic


class FACMACCritic(Critic):
    def __init__(self, scheme, rl, N):
        super().__init__(rl, N)

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
                 actions.contiguous().view(-1, self.n_actions)],
                dim=-1
            )
        if self.rl['nn_type'] == 'cnn':
            inputs = inputs.view(inputs.size()[0], 1, inputs.size()[1])

        x = F.relu(self.fc1(inputs))
        for i in range(len(self.layers)):
            if self.rl['nn_type'] == 'cnn' and i == self.rl['n_cnn_layers_critic'] - 1:
                x = nn.Flatten()(x)
            x = F.relu(self.layers[i](x))

        q = self.fc_out(x)

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
        if self.rl['nn_type'] == 'cnn':
            x = nn.Flatten()(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc_hidden_1(x))
        q = self.fc3(x)

        return q, hidden_state
