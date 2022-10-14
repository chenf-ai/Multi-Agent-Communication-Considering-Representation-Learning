import torch
from torch import nn
from torch.nn import functional as F


class ObVAEEnc(nn.Module):

    def __init__(self,
                 input_shape,
                 output_shape,
                 latent_dim,
                 args,
                 enc_hidden_dims=None,
                 dec_hidden_dims=None,
                 **kwargs) -> None:
        super(ObVAEEnc, self).__init__()

        self.latent_dim = latent_dim
        self.args = args

        if enc_hidden_dims is None:
            enc_hidden_dims = args.vae_enc_hidden_dims
            dec_hidden_dims = args.vae_dec_hidden_dims

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
        
        self.fc_mu = nn.Linear(self.args.encoder_hidden_dim, latent_dim)
        self.fc_var = nn.Linear(self.args.encoder_hidden_dim, latent_dim)

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

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)

        return mu, log_var, h

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

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]

        理解参考：https://blog.csdn.net/JohnJim0/article/details/110230703（zlc）
        这里用高斯分布采样，是为了加一些噪声吗？
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def z_forward(self, input, encoder_hidden_state, **kwargs):
        mu, log_var, h = self.encode(input, encoder_hidden_state)
        z = self.reparameterize(mu, log_var)
        return z, h

    def forward(self, input, encoder_hidden_state, **kwargs):
        mu, log_var, h = self.encode(input, encoder_hidden_state)
        z = self.reparameterize(mu, log_var)
        return  self.decode(z), input, mu, log_var, h

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
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['kld_weight']
        
        # 只能理解这是正常的VAE的loss，这跟算法pipline里面的VAE前半部分是一个意思？
        recons_loss = torch.mean((recons - output) ** 2, dim=-1) 

        # 这里实际是按一维高斯分布求解吗？如果这样，log_var表示的是logσ^2？（log(标准差））
        # 参考理解：https://hsinjhao.github.io/2019/05/22/KL-DivergenceIntroduction/
        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        input space.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)

        这个是用来干什么的？
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input, returns the input
        :param x: (Tensor) [B x D_IN]
        :return: (Tensor) [B x D_IN]
        """

        return self.forward(x)[0]


if __name__ == "__main__":
    # TODO: test vae module
    pass