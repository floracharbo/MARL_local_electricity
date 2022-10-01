# adapted from
# https://github.com/oxwhirl/facmac

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FACMACCritic(nn.Module):
    def __init__(self, scheme, rl):
        super(FACMACCritic, self).__init__()
        self.rl = rl
        self.n_actions = rl['dim_actions']
        self.n_agents = rl['n_agents']
        self.cuda_available = True if th.cuda.is_available() else False
        device = th.device("cuda") if self.cuda_available else th.device("cpu")

        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, rl['rnn_hidden_dim'])
        self.fc1.to(device)
        self.fc2 = nn.Linear(rl['rnn_hidden_dim'], rl['rnn_hidden_dim'])
        self.fc2.to(device)
        self.fc3 = nn.Linear(rl['rnn_hidden_dim'], 1)
        self.fc3.to(device)

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

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape


class FACMACDiscreteCritic(nn.Module):
    def __init__(self, scheme, rl):
        super(FACMACDiscreteCritic, self).__init__()
        self.rl = rl
        self.n_actions = scheme["actions_onehot"]["vshape"][0]
        self.n_agents = rl['n_agents']

        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, rl['rnn_hidden_dim'])
        self.fc2 = nn.Linear(rl['rnn_hidden_dim'], rl['rnn_hidden_dim'])
        self.fc3 = nn.Linear(rl['rnn_hidden_dim'], 1)

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

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape
