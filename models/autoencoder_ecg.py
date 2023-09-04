import torch
from torch import nn

from models.layers import ECG_Decoder, ECG_Encoder


class AutoencoderECG(nn.Module):
    def __init__(self, embedding_size: int, input_size: tuple):
        super(AutoencoderECG, self).__init__()
        self.input_size = input_size
        encoder_output_size = embedding_size
        self.encoder = ECG_Encoder(encoder_output_size, input_size=input_size)
        self.decoder = ECG_Decoder(embedding_size, input_size=input_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x.view(-1, *self.input_size))
        return self.decode(z)


class VariationalAutoencoderECG(nn.Module):
    def __init__(self, encoder_output_size: int, embedding_size: int, input_size: tuple):
        super(VariationalAutoencoderECG, self).__init__()
        self.input_size = input_size

        # Encoder output size
        self.encoder_output_size = encoder_output_size
        self.encoder = ECG_Encoder(
            input_size=input_size,
            output_size=encoder_output_size,
        )

        # From encoder to embedding
        self.var = nn.Linear(encoder_output_size, embedding_size)
        self.mu = nn.Linear(encoder_output_size, embedding_size)
        self.decoder = ECG_Decoder(embedding_size, input_size=input_size)

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
