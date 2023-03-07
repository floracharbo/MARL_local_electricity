# adapted from
# https://github.com/oxwhirl/facmac

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, rl):
        super(RNNAgent, self).__init__()

        self.rl = rl
        self.fc1 = nn.Linear(input_shape, rl['rnn_hidden_dim'])
        self.rnn = nn.GRUCell(self.rl['rnn_hidden_dim'],
                              self.rl['rnn_hidden_dim'])
        self.fc2 = nn.Linear(self.rl['rnn_hidden_dim'], self.rl['dim_actions'])

    def init_hidden(self):
        # make hidden states on same device as model
        if self.rl['init_weights_zero']:
            return self.fc1.weight.new(1, self.rl['rnn_hidden_dim']).zero_()
        else:
            return self.fc1.weight.new(1, self.rl['rnn_hidden_dim']).normal_(
                mean=0.0, std=self.rl['hyper_initialization_nonzeros']
            )

    def forward(self, inputs, hidden_state, actions=None):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rl['rnn_hidden_dim'])
        h = self.rnn(x, h_in)
        actions = th.tanh(self.fc2(h))
        return {"actions": actions, "hidden_state": h}
