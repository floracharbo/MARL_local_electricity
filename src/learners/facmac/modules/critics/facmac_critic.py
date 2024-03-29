# adapted from
# https://github.com/oxwhirl/facmac


import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


class FACMACCritic(nn.Module):
    def __init__(self, scheme, rl, N):
        super(FACMACCritic, self).__init__()
        self.rl = rl
        for info in ['dim_actions', 'n_homes', 'pruning_rate']:
            setattr(self, info, rl[info])
        self.output_type = "q"
        self.N = N
        self.input_shape = scheme["obs"]["vshape"] + self.dim_actions
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
                [inputs.view(-1, self.input_shape - self.dim_actions),
                 actions.contiguous().view(-1, self.dim_actions)],
                dim=-1
            )
        if self.rl['nn_type_critic'] == 'cnn':
            inputs = inputs.view(inputs.size()[0], 1, inputs.size()[1])
        x = F.relu(self.fc1(inputs.type(th.float32)))
        for i in range(len(self.layers)):
            if self.rl['nn_type_critic'] == 'cnn' and i == self.rl['n_cnn_layers_critic'] - 1:
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

        self._prune()

    def _prune(self):
        if self.pruning_rate > 0:
            prune.random_unstructured(self.fc1, name="weight", amount=self.rl['pruning_rate'])
            prune.random_unstructured(self.fc_out, name="weight", amount=self.rl['pruning_rate'])

            # for layer in ['fc1', 'fc_out']:
            #     prune.random_unstructured(self.__dict__[layer], name="weight", amount=self.rl['pruning_rate'])
            for i in range(len(self.layers)):
                if not isinstance(self.layers[i], nn.Dropout):
                    prune.random_unstructured(self.layers[i], name="weight", amount=self.rl['pruning_rate'])
