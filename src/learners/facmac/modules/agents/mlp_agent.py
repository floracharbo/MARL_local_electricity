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
        self.hidden = None

    def forward(self, inputs, hidden_state, actions=None, hidden=None):
        if self.rl['nn_type'] == 'cnn':
            inputs = inputs.view(inputs.size()[0], 1, inputs.size()[1])
        inputs = inputs.to(self.device)
        if self.rl['nn_type'] in ['lstm', 'rnn']:
            batch_size = int(inputs.size()[0] / self.n_agents)
            inputs = inputs.view(batch_size, self.n_agents, inputs.size()[1])
            if self.rl['nn_type'] == 'lstm':
                if self.hidden is None:
                    self.hidden = (th.zeros(1, batch_size, self.rl['rnn_hidden_dim']),
                                   th.zeros(1, batch_size, self.rl['rnn_hidden_dim']))
                x, self.hidden = self.fc1(inputs, self.hidden)
            else:
                x, h_n = self.fc1(inputs)
            x = F.relu(x)
        else:
            x = F.relu(self.fc1(inputs))
        for i in range(len(self.layers)):
            if self.rl['nn_type'] == 'cnn' and i == self.rl['n_cnn_layers'] - 1:
                x = nn.Flatten()(x)
            x = F.relu(self.layers[i](x))

        q = self.fc_out(x)
        # if self.agent_return_logits:
        if self.rl['learner'] == 'facmac_learner_discrete':
            # action_ps = F.gumbel_softmax(self.fc_out(x), tau=1, hard=False)
            actions = q
            # actions = F.gumbel_softmax(action_ps, tau=1, hard=False)

            # a = self.fc_out(x).view(self.rl['n_homes'], self.rl['dim_actions_1'], self.rl['n_discrete_actions'])
            # b = F.gumbel_softmax(a, tau=1, hard=False)
            # a.size()
            # Out[5]: torch.Size([10, 3, 10])
            # b.size()
            # Out[6]: torch.Size([10, 3, 10])
            # b[0, 0].sum()
            # Out[7]: tensor(1., grad_fn= < SumBackward0 >)
            # c = b.argmax(dim=2)
        else:
            actions = th.tanh(q)

        return {"actions": actions, "hidden_state": hidden_state, "Q": q}
