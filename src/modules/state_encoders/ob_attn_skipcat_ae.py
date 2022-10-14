import torch
from torch import nn
from torch.nn import functional as F


class ObAttnSkipCatAEEnc(nn.Module):

    def __init__(self,
                 input_shape,
                 output_shape,
                 n_agents,
                 latent_dim,
                 args,
                 enc_hidden_dims=None,
                 dec_hidden_dims=None,
                 **kwargs) -> None:
        super(ObAttnSkipCatAEEnc, self).__init__()

        # real latent dim here should be latent_dim*n_agents
        self.latent_dim = latent_dim
        self.n_agents = n_agents
        self.args = args

        if enc_hidden_dims is None:
            enc_hidden_dims = args.ae_enc_hidden_dims + [self.latent_dim]
            dec_hidden_dims = args.ae_dec_hidden_dims + [output_shape]

        self.query = nn.Linear(input_shape, self.args.attn_embed_dim)
        self.key = nn.Linear(input_shape, self.args.attn_embed_dim)
        self.value = nn.Linear(input_shape, self.args.encoder_hidden_dim)

        self.skip_layer = nn.Linear(input_shape, self.args.encoder_hidden_dim)

        if self.args.encoder_use_rnn:
            self.encoder_rnn = nn.GRUCell(self.args.encoder_hidden_dim, self.args.encoder_hidden_dim)
        else:
            self.encoder_rnn = nn.Linear(self.args.encoder_hidden_dim, self.args.encoder_hidden_dim)
        
        modules = []
        last_h_dim = self.args.encoder_hidden_dim * 2
        for i, h_dim in enumerate(enc_hidden_dims):
            if i == len(enc_hidden_dims) - 1:
                modules.append(
                    nn.Sequential(
                        nn.Linear(last_h_dim, h_dim),
                    )
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Linear(last_h_dim, h_dim),
                        nn.ReLU(),
                    )
                )
            last_h_dim = h_dim
        self.encoder = nn.Sequential(*modules)
        
        modules = []
        last_h_dim = self.latent_dim * self.n_agents
        for i, h_dim in enumerate(dec_hidden_dims):
            if i == len(dec_hidden_dims) - 1:
                modules.append(
                    nn.Sequential(
                        nn.Linear(last_h_dim, h_dim),
                    )
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Linear(last_h_dim, h_dim),
                        nn.ReLU(),
                    )
                )
            last_h_dim = h_dim
        self.decoder = nn.Sequential(*modules)
                 
    def encode(self, inputs, encoder_hidden_state):
        # input.shape: [bs*n_agents, input_shape]
        bs = inputs.shape[0] // self.n_agents

        def self_attention(query, key, value, attn_dim):
            # query.shape: [bs, n_agents, attn_embed_dim]
            # key.shape: [bs, attn_embed_dim, n_agents]
            # value.shape: [bs, n_agents, hidden_dim]
            energy = torch.bmm(query, key/(attn_dim ** (1/2)))
            score = F.softmax(energy, dim=-1)   # [bs, n_agents, n_agents]
            attn_out = torch.bmm(score, value)  # [bs, n_agents, hidden_dim]
            return attn_out
        
        query = self.query(inputs).reshape(bs, self.n_agents, self.args.attn_embed_dim)
        key = self.key(inputs).reshape(bs, self.n_agents, self.args.attn_embed_dim).permute(0, 2, 1)
        value = self.value(inputs).reshape(bs, self.n_agents, self.args.encoder_hidden_dim)
        # attn_out.shape: [bs, n_agents, hidden_dim]
        attn_out = self_attention(query, key, value, self.args.attn_embed_dim).reshape(bs*self.n_agents, self.args.encoder_hidden_dim)

        # h_in: [bs*n_agents, hidden_dim]
        h_in = encoder_hidden_state.reshape(-1, self.args.encoder_hidden_dim)
        if self.args.encoder_use_rnn:
            h = self.encoder_rnn(attn_out, h_in)
        else:
            h = F.relu(self.encoder_rnn(attn_out))

        # skip_out.shape: [bs*n_agents, encoder_hidden_dim]
        skip_out = self.skip_layer(inputs)
        encoder_input = torch.cat([h, skip_out], dim=-1)

        # z.shape: [bs*n_agents, latent_dim]
        z = self.encoder(encoder_input).reshape(bs, self.n_agents*self.latent_dim)
        return z, h

    def decode(self, z):
        # z.shape: [bs, n_agents*latent_dim]
        result = self.decoder(z)
        return result

    def forward(self, inputs, encoder_hidden_state, **kwargs):
        # inputs.shape: [bs*n_agents, input_shape]
        z, h = self.encode(inputs, encoder_hidden_state)        
        return self.decode(z), input, z, h

    def loss_function(self,
                      *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        !!!
            We suppose input.shape is in form of [batch_size, dim]
        !!!
        """        
        recons = args[0] # decoder(z)
        output = args[1]  # decoder要还原的内容，即state
        
        # 只能理解这是正常的VAE的loss，这跟算法pipline里面的VAE前半部分是一个意思？
        recons_loss = torch.mean((recons - output) ** 2, dim=-1) 

        loss = recons_loss
        return {'loss': loss}
        

if __name__ == "__main__":
    # TODO: test vae module
    pass