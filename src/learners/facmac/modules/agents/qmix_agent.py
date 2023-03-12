# adapted from
# https://github.com/oxwhirl/facmac

import torch.nn as nn
import torch.nn.functional as F

from src.learners.facmac.modules.agents.agent import Agent


class QMIXRNNAgent(nn.Module):
    def __init__(self, input_shape, rl, n_agents, N):
        super(QMIXRNNAgent, self).__init__()
        self.rl = rl
        self.n_agents = n_agents
        self.N = N
        self.fc1 = nn.Linear(input_shape, self.rl['rnn_hidden_dim'])
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

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rl['rnn_hidden_dim'])
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class FFAgent(Agent):
    def __init__(self, input_shape, rl):
        super(FFAgent, self).__init__(input_shape, rl)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        for i in range(len(self.fcs)):
            h = F.relu(self.fcs[i](x))
        q = self.fc_out(h)
        return q, h
