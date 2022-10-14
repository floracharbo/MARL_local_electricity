# adapted from
# https://github.com/oxwhirl/facmac

import torch as th
import torch.nn.functional as F
from learners.facmac.modules.agents.agent import Agent


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
