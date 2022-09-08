import torch.nn as nn
import torch.nn.functional as F


class QMIXRNNAgent(nn.Module):
    def __init__(self, input_shape, rl):
        super(QMIXRNNAgent, self).__init__()
        self.rl = rl

        self.fc1 = nn.Linear(input_shape, self.rl['rnn_hidden_dim'])
        self.rnn = nn.GRUCell(self.rl['rnn_hidden_dim'],
                              self.rl['rnn_hidden_dim'])
        self.fc2 = nn.Linear(self.rl['rnn_hidden_dim'], self.rl['dim_actions'])

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rl['rnn_hidden_dim']).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rl['rnn_hidden_dim'])
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class FFAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(FFAgent, self).__init__()
        self.args = args

        # Easiest to reuse rnn_hidden_dim variable
        self.fc1 = nn.Linear(input_shape, self.rl['rnn_hidden_dim'])
        self.fc2 = nn.Linear(self.rl['rnn_hidden_dim'],
                             self.rl['rnn_hidden_dim'])
        self.fc3 = nn.Linear(self.rl['rnn_hidden_dim'], self.rl['dim_actions'])

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rl['rnn_hidden_dim']).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        # h_in = hidden_state.reshape(-1, self.rl['rnn_hidden_dim'])
        h = F.relu(self.fc2(x))
        q = self.fc3(h)
        return q, h
