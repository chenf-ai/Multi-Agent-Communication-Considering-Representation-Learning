import torch.nn as nn
import torch.nn.functional as F
import torch as th


class TransitionModel(nn.Module):

    def __init__(self, args):
        super(TransitionModel, self).__init__()
        self.args = args

        self.action_embed = nn.Linear(args.n_actions, args.action_embed_dim)
        if args.latent_model_type == "squared":
            self.forward_net = nn.Linear(args.action_embed_dim * args.n_agents + args.state_repre_dim, args.state_repre_dim)
        elif args.latent_model_type == "kl":
            self.forward_net = nn.Linear(args.action_embed_dim * args.n_agents + args.state_repre_dim, args.state_repre_dim * 2)
        else:
            raise ValueError(f"Unknown latent model type: {args.latent_model_type}")

    def init_hidden(self):
        # make hidden states on same device as model
        pass

    def forward(self, state_repre, actions):
        # state_repre.shape: [batch_size, state_repre_dim]
        # actions.shape: [batch_size, seq_len, n_agents, n_actions]
        batch_size, seq_len, n_agents, _ = actions.shape
        # action_embed: [batch_size, seq_len, n_agents * action_embed_dim]
        action_embed = self.action_embed(actions).reshape(batch_size, seq_len, n_agents * self.args.action_embed_dim)
        forward_input = th.cat([state_repre, action_embed], dim=-1)
        pred_state_repre = self.forward_net(forward_input)
        return pred_state_repre