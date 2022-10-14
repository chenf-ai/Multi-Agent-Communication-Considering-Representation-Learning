import torch.nn as nn
import torch.nn.functional as F


class MLPEstimator(nn.Module):
    def __init__(self, input_shape, args):
        # Input_shape should be n_agents*obs_dim
        super(MLPEstimator, self).__init__()
        self.args = args

        # Define estimator function
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.est_use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.state_repre_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state=None):
        # inputs.shape: [batch_size, n_agents*obs_dim]
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.est_use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        estimated_repre = self.fc2(h)
        # estimated_repre.shape: [batch_size, state_repre_dim]
        return estimated_repre, h