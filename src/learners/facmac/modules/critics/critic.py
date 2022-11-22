import torch as th
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, rl, N):
        super(Critic, self).__init__()
        self.rl = rl
        self.n_actions = rl['dim_actions']
        self.n_homes = rl['n_homes']
        self.output_type = "q"
        self.N = N

    def set_up_network_layers(self, rl):
        self.cuda_available = True if th.cuda.is_available() else False
        device = th.device("cuda") if self.cuda_available else th.device("cpu")

        # Set up network layers
        if self.rl['nn_type'] == 'linear':
            self.fc1 = nn.Linear(self.input_shape, rl['rnn_hidden_dim'])
            self.fc2 = nn.Linear(rl['rnn_hidden_dim'], rl['rnn_hidden_dim'])
            self.fc3 = nn.Linear(rl['rnn_hidden_dim'], 1)
        elif self.rl['nn_type'] == 'cnn':
            self.fc1 = nn.Conv1d(1, rl['cnn_out_channels'], kernel_size=rl['cnn_kernel_size'])
            self.fc2 = nn.Linear((self.input_shape - 2) * rl['cnn_out_channels'], self.rl['rnn_hidden_dim'])
            self.fc3 = nn.Linear(rl['rnn_hidden_dim'], 1)

        self.fc1 = nn.DataParallel(self.fc1)
        self.fc2 = nn.DataParallel(self.fc2)
        self.fc3 = nn.DataParallel(self.fc3)
        self.fc1.to(device)
        self.fc2.to(device)
        self.fc3.to(device)



