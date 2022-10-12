import torch as th


class Learner():
    def __init__(self):
        self.mixer = None

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
        if self.args.verbose:
            self.logger.console_logger.info(
                f"Updated all target networks (soft update tau={tau})")

    def _get_target_actions_batch(self, batch, t_env):
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.select_actions(
                batch, t_ep=t, t_env=t_env, test_mode=True)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat over time

        return target_mac_out


    def _append_chosen_action_qvals(self, actions, batch, agent_outs, chosen_action_qvals, t):
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
