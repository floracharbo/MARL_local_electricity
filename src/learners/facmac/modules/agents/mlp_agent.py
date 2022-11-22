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
        print(f"self.device {self.device}")
        print(f"inputs.is_cuda {inputs.is_cuda}")
        print(f"next(self.fc1.parameters()).is_cuda {next(self.fc1.parameters()).is_cuda}")
        x = F.relu(self.fc1(inputs))
        if self.rl['nn_type'] == 'cnn':
            x = nn.Flatten()(x)
        for i in range(self.rl['n_hidden_layers']):
            x = x.cuda() if self.cuda_available else x
            x = F.relu(self.fcs[i](x))
        if self.agent_return_logits:
            actions = self.fc_out(x)
        else:
            actions = th.tanh(self.fc_out(x))
        return {"actions": actions, "hidden_state": hidden_state}
