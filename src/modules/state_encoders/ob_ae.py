import torch
from torch import nn
from torch.nn import functional as F


class ObAEEnc(nn.Module):

    def __init__(self,
                 input_shape,
                 output_shape,
                 latent_dim,
                 args,
                 enc_hidden_dims=None,
                 dec_hidden_dims=None,
                 **kwargs) -> None:
        super(ObAEEnc, self).__init__()

        self.latent_dim = latent_dim
        self.args = args

        if enc_hidden_dims is None:
            enc_hidden_dims = args.ae_enc_hidden_dims
            dec_hidden_dims = args.ae_dec_hidden_dims
            
        # Build Encoder
        modules = []
        last_h_dim = input_shape
        for h_dim in enc_hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(last_h_dim, h_dim),
                    nn.LeakyReLU(),
                )
            )
            last_h_dim = h_dim
        self.encoder = nn.Sequential(*modules)
        if self.args.encoder_use_rnn:
            self.encoder_rnn = nn.GRUCell(enc_hidden_dims[-1], self.args.encoder_hidden_dim)
        else:
            self.encoder_rnn = nn.Linear(enc_hidden_dims[-1], self.args.encoder_hidden_dim)
        
        self.fc_repr = nn.Linear(self.args.encoder_hidden_dim, latent_dim)
        
        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, dec_hidden_dims[0])
        modules = []
        last_h_dim = dec_hidden_dims[0]
        for h_dim in dec_hidden_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.Linear(last_h_dim, h_dim),
                    nn.LeakyReLU(),
                )
            )
            last_h_dim = h_dim
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Linear(dec_hidden_dims[-1], output_shape)
        
    def encode(self, input, encoder_hidden_state):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x D_IN]
        :return: (Tensor) List of latent codes
        """
        x = self.encoder(input)
        h_in = encoder_hidden_state.reshape(-1, self.args.encoder_hidden_dim)
        if self.args.encoder_use_rnn:
            h = self.encoder_rnn(x, h_in)
        else:
            h = F.relu(self.encoder_rnn(x))
        # Get final state repr
        z = self.fc_repr(h)
        return z, h

    def decode(self, z):
        """
        Maps the given latent codes
        onto the input space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x D_IN]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input, encoder_hidden_state, **kwargs):
        z, h = self.encode(input, encoder_hidden_state)        
        return [self.decode(z), input, z, h]

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