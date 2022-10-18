import copy
from src.learners.facmac.modules.critics.maddpg import MADDPGCritic
from src.learners.facmac.modules.mixers.qmix import QMixer
from src.learners.facmac.modules.mixers.vdn import VDNMixer

import torch as th
from torch.optim import Adam, RMSprop


class Learner():
    def __init__(self, mac, rl, scheme):
        self.mixer = None
        self.critic = None
        self.rl = rl
        self.n_agents = rl['n_homes']
        self.n_actions = rl['dim_actions']
        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())
        self.named_params = dict(mac.named_parameters())
        self.cuda_available = True if th.cuda.is_available() else False

        if self.__name__[0:6] == 'MADDPG':
            self.maddpg_init(mac, scheme, rl)
        elif self.__name__ in ['FACMACLearner', 'CQLearner']:
            self.init_mixer(rl)

    def init_mixer(self, rl):
        if rl['mixer'] is not None \
                and self.n_agents > 1:
            # if just 1 agent do not mix anything
            if rl['mixer'] == "vdn":
                self.mixer = VDNMixer()
            elif rl['mixer'] == "qmix":
                self.mixer = QMixer(rl)
            # elif rl['mixer'] == "vdn-s":
            #     self.mixer = VDNState(rl)
            # elif rl['mixer'] == "qmix-nonmonotonic":
            #     self.mixer = QMixerNonmonotonic(rl)
            else:
                raise ValueError(f"Mixer {rl['mixer']} not recognised.")

    def maddpg_init(self, mac, scheme, rl):
        self.agent_params = list(mac.parameters())

        self.critic = MADDPGCritic(scheme, rl)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        if self.rl['optimizer'] == "rmsprop":
            self.agent_optimiser = RMSprop(params=self.agent_params,
                                           lr=rl['lr'],
                                           alpha=self.rl['optim_alpha'],
                                           eps=self.rl['optim_eps'])
        elif self.rl['optimizer'] == "adam":
            self.agent_optimiser = Adam(params=self.agent_params,
                                        lr=rl['lr'],
                                        eps=self.rl['optimizer_epsilon'])
        else:
            raise Exception(f"unknown optimizer {self.rl['optimizer']}")

        if self.rl['optimizer'] == "rmsprop":
            self.critic_optimiser = RMSprop(params=self.critic_params,
                                            lr=self.rl['facmac']['critic_lr'],
                                            alpha=self.rl['optim_alpha'],
                                            eps=self.rl['optim_eps'])
        elif self.rl['optimizer'] == "adam":
            self.critic_optimiser = Adam(params=self.critic_params,
                                         lr=self.rl['facmac']['critic_lr'],
                                         eps=self.rl['optimizer_epsilon'])
        else:
            raise Exception(f"unknown optimizer {self.rl['optimizer']}")

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(),
                                       self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau)
                                    + param.data * tau)

        for target_param, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau)
                                    + param.data * tau)

        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(),
                                           self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau)
                                        + param.data * tau)

    def _get_target_actions_batch(self, batch, t_env):
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.select_actions(
                batch, t_ep=t, t_env=t_env, test_mode=True)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat over time

        return target_mac_out

    def _append_chosen_action_qvals(
            self, actions, batch, agent_outs, chosen_action_qvals, t
    ):
        for idx in range(self.n_agents):
            tem_joint_act = actions[:, t: t + 1].detach().clone().view(
                batch.batch_size, -1, self.n_actions)
            tem_joint_act[:, idx] = agent_outs[:, idx]
            q, _ = self.critic(self._build_inputs(batch, t=t),
                               tem_joint_act)
            chosen_action_qvals.append(q.view(batch.batch_size, -1, 1))

        return chosen_action_qvals

    def cuda(self, device="cuda:0"):
        self.mac.cuda(device=device)
        self.target_mac.cuda(device=device)
        if self.critic is not None:
            self.critic.cuda(device=device)
            self.target_critic.cuda(device=device)
        if self.mixer is not None:
            self.mixer.cuda(device=device)
            self.target_mixer.cuda(device=device)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def compute_grad_loss(self, q_taken, targets, mask):
        td_error = (q_taken - targets.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.critic_optimiser.zero_grad()
        loss.backward()

        return masked_td_error, loss

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))

        th.save(self.agent_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    f"{path}/mixer.th",
                    map_location=lambda storage, loc: storage)
            )
        self.agent_optimiser.load_state_dict(
            th.load(
                f"{path}/opt.th",
                map_location=lambda storage, loc: storage)
        )
