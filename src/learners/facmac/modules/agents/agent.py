
import torch as th
import torch.nn as nn


class Agent(nn.Module):
    def __init__(self, input_shape, rl, n_agents, N):
        super(Agent, self).__init__()
        self.rl = rl
        self.n_agents = n_agents
        self.N = N
        self.cuda_available = True if th.cuda.is_available() else False
        self.device = th.device("cuda") if self.cuda_available else th.device("cpu")
        if self.rl['nn_type'] == 'linear':
            self.fc1 = nn.Linear(input_shape, self.rl['rnn_hidden_dim'])
            self.layers = []
            for i in range(self.rl["n_hidden_layers"]):
                self.layers.append(
                    nn.Linear(self.rl['rnn_hidden_dim'], self.rl['rnn_hidden_dim'])
                )

        elif self.rl['nn_type'] == 'cnn':
            self.fc1 = nn.Conv1d(1, rl['cnn_out_channels'], kernel_size=rl['cnn_kernel_size'])
            self.layers = nn.ModuleList([])
            if self.rl['n_cnn_layers'] > 1:
                self.layers.extend(
                    [
                        nn.Conv1d(
                            rl['cnn_out_channels'], rl['cnn_out_channels'],
                            kernel_size=rl['cnn_kernel_size']
                        )
                        for _ in range(self.rl['n_cnn_layers'] - 1)
                    ]
                )
            self.layers.append(
                nn.Linear(
                    (input_shape - rl['cnn_kernel_size'] + 1) * rl['cnn_out_channels'],
                    self.rl['rnn_hidden_dim']
                )
            )
            if self.rl['n_hidden_layers'] > 1:
                self.layers.extend(
                    [
                        nn.Linear(self.rl['rnn_hidden_dim'], self.rl['rnn_hidden_dim'])
                        for _ in range(self.rl['n_hidden_layers'] - 1)
                    ]
                )

        elif self.rl['nn_type'] in ['rnn', 'lstm']:
            if self.rl['nn_type'] == 'lstm':
                self.fc1 = nn.LSTM(
                    input_shape, rl['rnn_hidden_dim'], num_layers=rl['num_layers_lstm'],
                    batch_first=True
                )
            elif self.rl['nn_type'] == 'rnn':
                self.fc1 = nn.RNN(
                    input_shape, rl['rnn_hidden_dim'], num_layers=rl['num_layers_rnn'],
                    batch_first=True
                )

            self.layers = nn.ModuleList(
                [nn.Linear(self.rl['rnn_hidden_dim'], self.rl['rnn_hidden_dim'])]
            )
            for i in range(self.rl['n_hidden_layers'] - 1):
                self.layers.append(
                    nn.Linear(self.rl['rnn_hidden_dim'], self.rl['rnn_hidden_dim'])
                )

        self.fc_out = nn.Linear(
            self.rl['rnn_hidden_dim'], self.rl['dim_actions']
        )

        self._gpu_parallelisation()
        self._layers_to_device()
        self.agent_return_logits = self.rl["agent_return_logits"]

    def _layers_to_device(self):
        self.fc1.to(self.device)
        for i in range(len(self.layers)):
            self.layers[i].to(self.device)
        self.fc_out.to(self.device)

    def _gpu_parallelisation(self):
        if self.rl['data_parallel']:
            self.fc1 = nn.DataParallel(self.fc1)
            self.fc_out = nn.DataParallel(self.fc_out)
            for i in range(len(self.layers)):
                self.layers[i] = nn.DataParallel(self.layers[i])

    def init_hidden(self):
        # make hidden states on same device as model
        if self.rl['data_parallel']:
            return self.fc1.module.weight.new(1, self.rl['rnn_hidden_dim']).zero_()
        elif self.rl['nn_type'] in ['lstm', 'rnn']:
            self.hidden = None
            return (
                self.fc1.weight_hh_l0.new(1, self.rl['rnn_hidden_dim']).zero_(),
                self.fc1.weight_ih_l0.new(1, self.rl['rnn_hidden_dim']).zero_()
            )
        else:
            return self.fc1.weight.new(1, self.rl['rnn_hidden_dim']).zero_()
