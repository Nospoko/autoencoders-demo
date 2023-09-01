import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F

from models.layers import CNN_Decoder, CNN_Encoder


class VariationalAutoencoder(nn.Module):
    model_type: str = "VAE"

    def __init__(self, output_size: int, embedding_size: int, input_size: tuple):
        super(VariationalAutoencoder, self).__init__()
        self.input_size = input_size

        # Encoder output size
        self.output_size = output_size
        self.encoder = CNN_Encoder(output_size=output_size, input_size=input_size)

        # From encoder to embedding
        self.var = nn.Linear(output_size, embedding_size)
        self.mu = nn.Linear(output_size, embedding_size)
        self.decoder = CNN_Decoder(embedding_size, input_size=input_size)

    def encode(self, x):
        x = self.encoder(x)
        return self.mu(x), self.var(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.contiguous().view(-1, *self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()

    def forward(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")

        # KL divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
