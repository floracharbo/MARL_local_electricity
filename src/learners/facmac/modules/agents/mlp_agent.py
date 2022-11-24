# adapted from https://github.com/oxwhirl/facmac
import torch as th
import torch.nn.functional as F
from torch import nn

from src.learners.facmac.modules.agents.agent import Agent


class MLPAgent(Agent):
    def __init__(self, input_shape, rl, n_agents, N):
        super(MLPAgent, self).__init__(input_shape, rl, n_agents, N)
        self.n_agents = n_agents
        self.N = N

    def forward(self, inputs, hidden_state, actions=None):
        if self.rl['nn_type'] == 'cnn':
            inputs = inputs.view(inputs.size()[0], 1, inputs.size()[1])
        inputs = inputs.to(self.device)
        x = F.relu(self.fc1(inputs))
        if self.rl['nn_type'] == 'cnn':
            x = nn.Flatten()(x)
        for i in range(len(self.fcs)):
            x = F.relu(self.fcs[i](x))
        # x = F.relu(self.fc_hidden_1(x))
        # for i in range(len(self.fcs)):
        #     if self.rl['nn_type'] == 'cnn' and i == self.rl['n_cnn_layers'] - 1:
        #         x = nn.Flatten()(x)
        #     x = x.cuda() if self.cuda_available else x
        #     x = F.relu(self.fcs[i](x))
        if self.agent_return_logits:
            actions = self.fc_out(x)
        else:
            actions = th.tanh(self.fc_out(x))

        return {"actions": actions, "hidden_state": hidden_state}
