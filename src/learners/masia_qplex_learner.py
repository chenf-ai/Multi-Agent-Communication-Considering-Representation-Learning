import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.dmaq_general import DMAQer
from modules.mixers.dmaq_qatten import DMAQ_QattenMixer
import torch as th
import torch.nn.functional as F
from torch.optim import Adam
from components.standarize_stream import RunningMeanStd
import os

class MASIALearner:
    def __init__(self, mac, latent_model, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.latent_model = latent_model
        self.logger = logger

        if not self.args.rl_signal:
            assert 0, "Must use rl signal in this method !!!"
            self.params = list(mac.rl_parameters())
        else:
            self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "dmaq":
                self.mixer = DMAQer(args)
            elif args.mixer == "dmaq_qatten":
                self.mixer = DMAQ_QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if self.args.use_latent_model:
            # use_latent_model means use_spr
            self.params += list(latent_model.parameters())

        self.optimiser = Adam(params=self.params, lr=args.lr)
                
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        self.n_actions = self.args.n_actions

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def repr_train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # states.shape: [batch_size, seq_len, state_dim]
        states = batch["state"]
        # actions.shape: [batch_size, seq_len, n_agents, 1]
        actions_onehot = batch["actions_onehot"]
        rewards = batch["reward"]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # go through vae
        recons, z = [], []
        self.mac.init_hidden(batch.batch_size)  # useless in current version
        for t in range(batch.max_seq_length):
            recons_t, _, z_t = self.mac.vae_forward(batch, t)
            recons.append(recons_t)
            z.append(z_t)
        # recons.shape: [batch_size, seq_len, state_repre_dim]
        recons = th.stack(recons, dim=1)  # Concat over time
        z = th.stack(z, dim=1)

        bs, seq_len  = states.shape[0], states.shape[1]
        loss_dict = self.mac.agent.encoder.loss_function(recons.reshape(bs*seq_len, -1), states.reshape(bs*seq_len, -1))
        vae_loss = loss_dict["loss"].reshape(bs, seq_len, 1)
        mask = mask.expand_as(vae_loss)
        masked_vae_loss = (vae_loss * mask).sum() / mask.sum()

        if self.args.use_latent_model:
            # Compute target z first
            target_projected = []
            with th.no_grad():
                self.mac.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    target_projected_t = self.mac.target_transform(batch, t)
                    target_projected.append(target_projected_t)
            target_projected = th.stack(target_projected, dim=1)  # Concat over time, shape: [bs, seq_len, spr_dim]

            curr_z = z
            # Do final vector prediction
            predicted_f = self.mac.agent.online_projection(curr_z)   # [bs, seq_len, spr_dim]
            tot_spr_loss = self.compute_spr_loss(predicted_f, target_projected, mask)
            if self.args.use_rew_pred:
                predicted_rew = self.latent_model.predict_reward(curr_z)   # [bs, seq_len, 1]
                tot_rew_loss = self.compute_rew_loss(predicted_rew, rewards, mask)
            for t in range(self.args.pred_len):
                # do transition model forward
                curr_z = self.latent_model(curr_z, actions_onehot[:, t:])[:, :-1]
                # Do final vector prediction
                predicted_f = self.mac.agent.online_projection(curr_z)  # [bs, seq_len, spr_dim]
                tot_spr_loss += self.compute_spr_loss(predicted_f, target_projected[:, t+1:], mask[:, t+1:])
                if self.args.use_rew_pred:
                    predicted_rew = self.latent_model.predict_reward(curr_z)
                    tot_rew_loss += self.compute_rew_loss(predicted_rew, rewards[:, t+1:], mask[:, t+1:])
            
            if self.args.use_rew_pred:
                repr_loss = masked_vae_loss + self.args.spr_coef * tot_spr_loss + self.args.rew_pred_coef * tot_rew_loss
            else:
                repr_loss = masked_vae_loss + self.args.spr_coef * tot_spr_loss
        else:
            repr_loss = masked_vae_loss

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("repr_loss", repr_loss.item(), t_env)
            self.logger.log_stat("vae_loss", masked_vae_loss.item(), t_env)
            if self.args.use_latent_model:
                self.logger.log_stat("model_loss", tot_spr_loss.item(), t_env)
                if self.args.use_rew_pred:
                    self.logger.log_stat("rew_pred_loss", tot_rew_loss.item(), t_env)

        return repr_loss
    
    def compute_rew_loss(self, pred_rew, env_rew, mask):
        # pred_rew.shape: [bs, seq_len, 1]
        # mask.shape: [bs, seq_len, 1]
        mask = mask.squeeze(-1)
        rew_loss = F.mse_loss(pred_rew, env_rew, reduction="none").sum(-1)
        masked_rew_loss = (rew_loss * mask).sum() / mask.sum()
        return masked_rew_loss

    def compute_spr_loss(self, pred_f, target_f, mask):
        # pred_f.shape: [bs, seq_len, spr_dim]
        # mask.shape: [bs, seq_len, 1]
        mask = mask.squeeze(-1)
        spr_loss = F.mse_loss(pred_f, target_f, reduction="none").sum(-1)
        mask_spr_loss = (spr_loss * mask).sum() / mask.sum()
        return mask_spr_loss

    def rl_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, repr_loss):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            state_repr_t = self.mac.enc_forward(batch, t=t)
            if not self.args.rl_signal:
                state_repr_t = state_repr_t.detach()
            agent_outs = self.mac.rl_forward(batch, state_repr_t, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            state_repr_t = self.target_mac.enc_forward(batch, t=t)
            target_agent_outs = self.target_mac.rl_forward(batch, state_repr_t, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_next_actions = cur_max_actions.detach()

            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)).cuda()
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        else:
            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            if self.args.mixer == "dmaq_qatten":
                ans_chosen, q_attend_regs, head_entropies = \
                    self.mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv, _, _ = self.mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                      max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv
            else:
                ans_chosen = self.mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv = self.mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv

            if self.args.double_q:
                if self.args.mixer == "dmaq_qatten":
                    target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                         actions=cur_max_actions_onehot,
                                                         max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
                else:
                    target_chosen = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                   actions=cur_max_actions_onehot,
                                                   max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        if self.args.mixer == "dmaq_qatten":
            rl_loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
        else:
            rl_loss = (masked_td_error ** 2).sum() / mask.sum()
        # Compute tot loss
        tot_loss = rl_loss + self.args.repr_coef * repr_loss

        # Optimise
        self.optimiser.zero_grad()
        tot_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.mac.agent.momentum_update()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)
            self.mac.agent.momentum_update()
        
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("rl_loss", rl_loss.item(), t_env)
            self.logger.log_stat("tot_loss", tot_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Representation learning training
        repr_loss = self.repr_train(batch, t_env, episode_num)
        # RL training
        self.rl_train(batch, t_env, episode_num, repr_loss)

    def test_encoder(self, batch: EpisodeBatch):
        # states.shape: [batch_size, seq_len, state_dim]
        states = batch["state"]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # go through vae
        recons, z = [], []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            recons_t, _, z_t = self.mac.vae_forward(batch, t)
            recons.append(recons_t)
            z.append(z_t)
        # recons.shape: [batch_size, seq_len, state_repre_dim]
        recons = th.stack(recons, dim=1)
        z = th.stack(z, dim=1)

        encoder_result = {
            "recons": recons,
            "z": z,
            "states": states,
            "mask": mask,
        }
        th.save(encoder_result, os.path.join(self.args.encoder_result_direc, "result.pth"))

    def _update_targets_hard(self):
        # not quite good, but don't have bad effect
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        # not quite good, but don't have bad effect
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.latent_model.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.latent_model.state_dict(), "{}/latent_model.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.latent_model.load_state_dict(th.load("{}/latent_model.th".format(path), map_location=lambda storage, loc: storage))