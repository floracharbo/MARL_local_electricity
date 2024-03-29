# adapted from
# https://github.com/oxwhirl/facmac

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    def __init__(self, rl):
        super(QMixer, self).__init__()

        self.rl = rl
        self.n_agents = rl['n_homes']
        self.state_dim = int(np.prod(rl['state_shape']))
        self.cuda_available = True if th.cuda.is_available() else False
        device = th.device("cuda") if self.cuda_available else th.device("cpu")
        self.embed_dim = rl['mixing_embed_dim']
        self.q_embed_dim = rl['q_embed_dim']
        self.hyper_w_1 = nn.Linear(
            self.state_dim,
            self.embed_dim * self.n_agents * self.q_embed_dim)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        if rl['hypernet_layers'] > 1:
            assert self.rl['hypernet_layers'] == 2, \
                "Only 1 or 2 hypernet_layers is supported atm!"
            hypernet_embed = self.rl['hypernet_embed']

            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed,
                          self.embed_dim * self.n_agents * self.q_embed_dim))
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim))

        # Initialise the hyper networks with a fixed variance, if specified
        if self.rl['hyper_initialization_nonzeros'] > 0:
            std = self.rl['hyper_initialization_nonzeros'] ** -0.5

            self.hyper_w_1.weight.data.normal_(std=std)
            self.hyper_w_1.bias.data.normal_(std=std)
            self.hyper_w_final.weight.data.normal_(std=std)
            self.hyper_w_final.bias.data.normal_(std=std)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        if rl['initialise_positive_weights_hyper_b_1_bias']:
            th.nn.init.uniform_(self.hyper_b_1.bias, a=0.2, b=0.5)  # Adjust the range as needed

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

        if self.rl['gated']:
            self.gate = nn.Parameter(th.ones(size=(1,)) * 0.5)

        if self.rl['data_parallel']:
            self.hyper_b_1 = nn.DataParallel(self.hyper_b_1)
            self.hyper_w_1 = nn.DataParallel(self.hyper_w_1)
            self.hyper_w_final = nn.DataParallel(self.hyper_w_final)
            self.V = nn.DataParallel(self.V)
        self.hyper_b_1.to(device)
        self.hyper_w_1.to(device)
        self.hyper_w_final.to(device)
        self.V.to(device)

    def forward(self, agent_qs, states):
        states = states.cuda() if self.cuda_available else states
        agent_qs = agent_qs.cuda() if self.cuda_available else agent_qs
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents * self.q_embed_dim)

        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        if self.rl['mixer_relu']:
            b1 = F.relu(b1)

        w1 = w1.view(- 1, self.n_agents * self.q_embed_dim, self.embed_dim)
        b1 = b1.view(- 1, 1, self.embed_dim)

        # hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        hidden = th.bmm(agent_qs, w1) + b1
        if self.rl['mixer_relu']:
            hidden = F.elu(hidden)

        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Skip connections
        s = 0
        if self.rl['skip_connections']:
            s = agent_qs.sum(dim=2, keepdim=True)

        if self.rl['gated']:
            y = th.bmm(hidden, w_final) * self.gate + v + s
        else:
            # Compute final output
            y = th.bmm(hidden, w_final) + v + s

        # Reshape and return
        q_tot = y.view(bs, -1, 1)

        return q_tot
