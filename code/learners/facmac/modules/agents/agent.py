import torch as th

import torch.nn.functional as F

import torch.nn as nn


class Agent(nn.Module):
    def __init__(self, input_shape, rl):
        super(Agent, self).__init__()
        self.rl = rl
        self.cuda_available = True if th.cuda.is_available() else False
        device = th.device("cuda") if self.cuda_available else th.device("cpu")

        self.fc1 = nn.Linear(input_shape, self.rl['rnn_hidden_dim'])
        self.fcs = []
        for i in range(self.rl["n_hidden_layers"]):
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


class MLPAgent(Agent):
    def __init__(self, input_shape, rl):
        super(MLPAgent, self).__init__(input_shape, rl)

    def forward(self, inputs, hidden_state, actions=None):
        x = F.relu(self.fc1(inputs))
        for i in range(self.rl['n_hidden_layers']):
            x = x.cuda() if self.cuda_available else x
            x = F.relu(self.fcs[i](x))
        if self.agent_return_logits:
            actions = self.fc_out(x)
        else:
            actions = th.tanh(self.fc_out(x))
        return {"actions": actions, "hidden_state": hidden_state}
