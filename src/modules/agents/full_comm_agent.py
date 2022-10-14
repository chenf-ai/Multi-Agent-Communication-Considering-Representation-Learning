import torch.nn as nn
import torch.nn.functional as F
import torch as th

from types import SimpleNamespace as SN


class FullCommAgent(nn.Module):
    """
        Full Communication Agent
    """
    def __init__(self, input_shape, args):
        super(FullCommAgent, self).__init__()
        self.args = args
        self.base_input_shape, action_input_shape = self._get_input_shape(input_shape, args)

        if args.big_net_agent:
            self.fc1 = nn.Sequential(
                nn.Linear(action_input_shape, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 32),
                nn.LeakyReLU(),
                nn.Linear(32, args.hidden_dim)
            )
        else:
            self.fc1 = nn.Linear(action_input_shape, args.hidden_dim)

        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc2.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # inputs.shape: [batch_size*n_agents, input_shape]
        action_inputs = self._build_inputs(inputs)

        # go through remaining networks
        x = F.relu(self.fc1(action_inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)

        return q, h

    def _build_inputs(self, inputs):
        # inputs.shape: [n_agents*batch_size, input_shape]
        bs = inputs.shape[0] // self.args.n_agents
        # extract different input components
        base_inputs = inputs[:, :self.base_input_shape]
        extra_inputs = inputs[:, self.base_input_shape:]
        # do full communication
        communicated = base_inputs.reshape(bs, self.args.n_agents, self.base_input_shape)
        # ----- noisy setting -----
        if self.args.noise_env and self.args.noise_type == 0:
            # TODO: not exactly right !!!
            noise = th.randn(*communicated.shape)
            communicated += noise.to(communicated.device)
        # repeat process; communicated.shape: [bs, n_agents, n_agents*base_input_shape]
        communicated = communicated.reshape(bs, self.args.n_agents * self.base_input_shape).unsqueeze(1).repeat(1, self.args.n_agents, 1)
        # reshape communicated
        communicated = communicated.reshape(bs, self.args.n_agents, self.args.n_agents, self.base_input_shape)
        # ----- noisy setting -----
        if self.args.noise_env and self.args.noise_type == 1:
            noise = th.randn(bs, self.args.n_agents, self.args.n_agents, communicated.shape[-1])
            mask = (1 - th.eye(self.args.n_agents, self.args.n_agents)).unsqueeze(0).repeat(bs, 1, 1).unsqueeze(-1)
            noise *= mask
            communicated += noise.to(communicated.device) 
        action_inputs = th.cat([communicated.reshape(bs * self.args.n_agents, self.args.n_agents * self.base_input_shape), extra_inputs], dim=-1)
        # action_inputs.shape: [batch_size*n_agents, input_shape]
        return action_inputs

    def _get_input_shape(self, input_shape, args):
        base_input_shape = input_shape
        if self.args.obs_last_action:
            base_input_shape -= args.n_actions
        if self.args.obs_agent_id:
            base_input_shape -= args.n_agents
        input_shape += (args.n_agents - 1) * base_input_shape
        return base_input_shape, input_shape