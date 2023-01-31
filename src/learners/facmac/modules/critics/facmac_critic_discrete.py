import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FACMACDiscreteCritic(nn.Module):
    def __init__(self, scheme, rl, N):
        super(FACMACDiscreteCritic, self).__init__()
        self.rl = rl
        self.n_actions = rl['dim_actions']
        self.n_agents = rl['n_homes']
        self.input_shape = scheme["obs"]["vshape"] + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        # self.set_up_network_layers(self.rl)

        # Set up network layers
        self.layers = nn.ModuleList([])
        self.fc1 = nn.Linear(self.input_shape, self.rl['rnn_hidden_dim'])
        if self.rl['n_hidden_layers_critic'] > 0:
            self.layers.extend(
                [
                    nn.Linear(self.rl['rnn_hidden_dim'], self.rl['rnn_hidden_dim'])
                    for _ in range(self.rl['n_hidden_layers_critic'])
                ]
            )
        self.fc_out = nn.Linear(self.rl['rnn_hidden_dim'], 1)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.reshape(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        x = F.relu(self.fc1(inputs))
        for i in range(len(self.layers)):
            if self.rl['nn_type'] == 'cnn' and i == self.rl['n_cnn_layers_critic'] - 1:
                x = nn.Flatten()(x)
            x = F.relu(self.layers[i](x))

        q = self.fc_out(x)

        return q, hidden_state
