import torch as th
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, rl):
        super(Critic, self).__init__()
        self.rl = rl
        self.n_actions = rl['dim_actions']
        self.n_agents = rl['n_agents']
        self.output_type = "q"

    def set_up_network_layers(self, rl):
        self.cuda_available = True if th.cuda.is_available() else False
        device = th.device("cuda") if self.cuda_available else th.device("cpu")

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, rl['rnn_hidden_dim'])
        self.fc1.to(device)
        self.fc2 = nn.Linear(rl['rnn_hidden_dim'], rl['rnn_hidden_dim'])
        self.fc2.to(device)
        self.fc3 = nn.Linear(rl['rnn_hidden_dim'], 1)
        self.fc3.to(device)
