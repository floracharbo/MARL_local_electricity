import copy

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
            self.fcs = []
            for i in range(self.rl["n_hidden_layers"]):
                self.fcs.append(
                    nn.Linear(self.rl['rnn_hidden_dim'], self.rl['rnn_hidden_dim'])
                )

        elif self.rl['nn_type'] == 'cnn':
            self.fc1 = nn.Conv1d(1, rl['cnn_out_channels'], kernel_size=rl['cnn_kernel_size'])
            self.layers = nn.ModuleList([])
            if self.rl['n_cnn_layers'] > 1:
                self.layers.extend([nn.Conv1d(rl['cnn_out_channels'], rl['cnn_out_channels'], kernel_size=rl['cnn_kernel_size']) for _ in range(self.rl['n_cnn_layers'] - 1)])
            self.layers.append(nn.Linear((input_shape - rl['cnn_kernel_size'] + 1) * rl['cnn_out_channels'], self.rl['rnn_hidden_dim']))
            if self.rl['n_hidden_layers'] > 1:
                self.layers.extend([nn.Linear(self.rl['rnn_hidden_dim'], self.rl['rnn_hidden_dim']) for _ in range(self.rl['n_hidden_layers'] - 1)])

        elif self.rl['nn_type'] == 'lstm':
            self.fc1 = nn.LSTM(input_shape, rl['rnn_hidden_dim'], num_layers=rl['num_layers_lstm'])
            self.fcs = [nn.Linear((input_shape - rl['cnn_kernel_size'] + 1) * rl['cnn_out_channels'], self.rl['rnn_hidden_dim'])]
            for i in range(self.rl['n_hidden_layers'] - 1):
                self.fcs.append(
                    nn.Linear(self.rl['rnn_hidden_dim'], self.rl['rnn_hidden_dim'])
                )

        self.fc_out = nn.Linear(
            self.rl['rnn_hidden_dim'], self.rl['dim_actions']
        )

        self._gpu_parallelisation()
        self._layers_to_device()
        self.agent_return_logits = self.rl["agent_return_logits"]

    def _layers_to_device(self):
        # self.fc1.to(self.device)
        for i in range(len(self.layers)):
            self.layers[i].to(self.device)
        # self.fc2.to(self.device)
        self.fc_out.to(self.device)
        # if self.rl['nn_type_critic'] == 'cnn':
        #     if self.rl['n_cnn_layers_critic'] > 1:
        #         self.fc_kernel_2.to(self.device)
        #     if self.rl['n_cnn_layers_critic'] > 2:
        #         self.fc_kernel_3.to(self.device)
        # if self.rl['n_hidden_layers_critic'] > 1:
        #     self.fc_hidden_2.to(self.device)
        # if self.rl['n_hidden_layers_critic'] > 2:
        #     self.fc_hidden_3.to(self.device)

    def _gpu_parallelisation(self):
        if self.rl['data_parallel']:
            self.fc1 = nn.DataParallel(self.fc1)
            self.fc2 = nn.DataParallel(self.fc2)
            self.fc_out = nn.DataParallel(self.fc_out)
            if self.rl['nn_type_critic'] == 'cnn':
                if self.rl['n_cnn_layers_critic'] > 1:
                    self.fc_kernel_2 = nn.DataParallel(self.fc_kernel_2)
                if self.rl['n_cnn_layers_critic'] > 2:
                    self.fc_kernel_3 = nn.DataParallel(self.fc_kernel_3)
            if self.rl['n_hidden_layers_critic'] > 1:
                self.fc_hidden_2 = nn.DataParallel(self.fc_hidden_2)
            if self.rl['n_hidden_layers_critic'] > 2:
                self.fc_hidden_3 = nn.DataParallel(self.fc_hidden_3)

    def init_hidden(self):
        # make hidden states on same device as model
        if self.rl['data_parallel']:
            return self.fc1.module.weight.new(1, self.rl['rnn_hidden_dim']).zero_()
        else:
            return self.fc1.weight.new(1, self.rl['rnn_hidden_dim']).zero_()
