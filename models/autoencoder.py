from torch import nn

from models.layers import CNN_Decoder, CNN_Encoder


class Autoencoder(nn.Module):
    model_type: str = "AE"

    def __init__(self, cfg, input_size):
        super(Autoencoder, self).__init__()
        self.cfg = cfg
        self.input_size = input_size
        output_size = cfg.model.embedding_size
        self.encoder = CNN_Encoder(output_size, input_size=input_size)
        self.decoder = CNN_Decoder(cfg.model.embedding_size, input_size=input_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x.view(-1, *self.input_size))
        return self.decode(z)
