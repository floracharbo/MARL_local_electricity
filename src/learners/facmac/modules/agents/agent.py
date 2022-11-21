import torch as th
import torch.nn as nn


class Agent(nn.Module):
    def __init__(self, input_shape, rl, n_agents, N):
        super(Agent, self).__init__()
        self.rl = rl
        self.n_agents = n_agents
        self.N = N
        self.cuda_available = True if th.cuda.is_available() else False
        device = th.device("cuda") if self.cuda_available else th.device("cpu")

        if self.rl['nn_type'] == 'linear':
            self.fc1 = nn.Linear(input_shape, self.rl['rnn_hidden_dim'])
            self.fcs = []
            for i in range(self.rl["n_hidden_layers"]):
                self.fcs.append(
                    nn.Linear(self.rl['rnn_hidden_dim'], self.rl['rnn_hidden_dim'])
                )
        elif self.rl['nn_type'] == 'cnn':
            self.fc1 = nn.Conv1d(1, 1, kernel_size=3)
            self.fcs = []
            self.fcs.append(nn.Linear(self.N - 2, self.rl['rnn_hidden_dim']))
            for i in range(self.rl["n_hidden_layers"] - 1):
                self.fcs.append(
                    nn.Linear(self.rl['rnn_hidden_dim'], self.rl['rnn_hidden_dim'])
                )

        self.fcs[-1].to(device)
        self.fc_out = nn.Linear(
            self.rl['rnn_hidden_dim'], self.rl['dim_actions']
        )
        self.fc_out.to(device)
        self.agent_return_logits = self.rl["agent_return_logits"]

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rl['rnn_hidden_dim']).zero_()
