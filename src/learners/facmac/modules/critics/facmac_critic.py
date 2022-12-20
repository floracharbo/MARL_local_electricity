# adapted from
# https://github.com/oxwhirl/facmac


import torch.nn.functional as F
import torch as th
import torch.nn as nn


class FACMACCritic(nn.Module):
    def __init__(self, scheme, rl, N):
        super(FACMACCritic, self).__init__()
        self.rl = rl
        self.n_actions = rl['dim_actions']
        self.n_homes = rl['n_homes']
        self.output_type = "q"
        self.N = N
        self.input_shape = scheme["obs"]["vshape"] + self.n_actions
        self.hidden_states = None

        # Set up network layers
        self.set_up_network_layers(rl)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = inputs.cuda() if self.cuda_available else inputs
            actions = actions.cuda() if self.cuda_available else actions
            inputs = th.cat(
                [inputs.view(-1, self.input_shape - self.n_actions),
                 actions.contiguous().view(-1, self.n_actions)],
                dim=-1
            )
        if self.rl['nn_type'] == 'cnn':
            inputs = inputs.view(inputs.size()[0], 1, inputs.size()[1])

        x = F.relu(self.fc1(inputs))
        for i in range(len(self.layers)):
            if self.rl['nn_type'] == 'cnn' and i == self.rl['n_cnn_layers_critic'] - 1:
                x = nn.Flatten()(x)
            x = F.relu(self.layers[i](x))

        q = self.fc_out(x)

        return q, hidden_state

    def set_up_network_layers(self, rl):
        self.cuda_available = True if th.cuda.is_available() else False
        device = th.device("cuda") if self.cuda_available else th.device("cpu")
        self.layers = nn.ModuleList([])

        # Set up network layers
        if self.rl['nn_type_critic'] == 'linear':
            self.fc1 = nn.Linear(self.input_shape, rl['rnn_hidden_dim'])
            if self.rl['n_hidden_layers_critic'] > 0:
                self.layers.extend(
                    [
                        nn.Linear(self.rl['rnn_hidden_dim'], self.rl['rnn_hidden_dim'])
                        for _ in range(self.rl['n_hidden_layers_critic'])
                    ]
                )

        elif self.rl['nn_type_critic'] == 'cnn':
            self.fc1 = nn.Conv1d(1, rl['cnn_out_channels'], kernel_size=rl['cnn_kernel_size'])
            if self.rl['n_cnn_layers_critic'] > 1:
                self.layers.extend(
                    [
                        nn.Conv1d(
                            self.rl['cnn_out_channels'],
                            self.rl['cnn_out_channels'],
                            kernel_size=rl['cnn_kernel_size']
                        )
                        for _ in range(self.rl['n_hidden_layers_critic'])
                    ]
                )
            self.layers.append(
                nn.Linear(
                    (self.input_shape - rl['cnn_kernel_size'] + 1) * rl['cnn_out_channels'],
                    self.rl['rnn_hidden_dim']
                )
            )

            if self.rl['n_hidden_layers_critic'] > 1:
                self.layers.extend(
                    [nn.Linear(self.rl['rnn_hidden_dim'], self.rl['rnn_hidden_dim'])
                     for _ in range(self.rl['n_hidden_layers_critic'] - 1)])

        self.fc_out = nn.Linear(rl['rnn_hidden_dim'], 1)

        if self.rl['data_parallel']:
            self.fc1 = nn.DataParallel(self.fc1)
            self.fc_out = nn.DataParallel(self.fc_out)
            for i in range(len(self.layers)):
                self.layers[i] = nn.DataParallel(self.layers[i])

        self.fc1.to(device)
        self.fc_out.to(device)
        for i in range(len(self.layers)):
            self.layers[i].to(device)
