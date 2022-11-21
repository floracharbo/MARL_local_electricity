import torch as th
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, rl):
        super(Critic, self).__init__()
        self.rl = rl
        self.n_actions = rl['dim_actions']
        self.n_homes = rl['n_homes']
        self.output_type = "q"

    def set_up_network_layers(self, rl):
        self.cuda_available = True if th.cuda.is_available() else False
        device = th.device("cuda") if self.cuda_available else th.device("cpu")

        # Set up network layers
        if self.rl['nn_type'] == 'linear':
            self.fc1 = nn.Linear(self.input_shape, rl['rnn_hidden_dim'])
            self.fc2 = nn.Linear(rl['rnn_hidden_dim'], rl['rnn_hidden_dim'])
            self.fc3 = nn.Linear(rl['rnn_hidden_dim'], 1)
        elif self.rl['nn_type'] == 'cnn':
            self.fc1 = nn.Conv1d(self.input_shape, self.rl['rnn_hidden_dim'], kernel_size=3)
            self.fc2 = nn.Linear(self.rl['rnn_hidden_dim'], self.rl['rnn_hidden_dim'])
            self.fc3 = nn.Linear(rl['rnn_hidden_dim'], 1)

        self.fc1.to(device)
        self.fc2.to(device)
        self.fc3.to(device)



