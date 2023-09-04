from torch import nn

from models.layers import CNN_Decoder, CNN_Encoder


class Autoencoder(nn.Module):
    def __init__(self, embedding_size: int, input_size: tuple):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        output_size = embedding_size
        self.encoder = CNN_Encoder(output_size, input_size=input_size)
        self.decoder = CNN_Decoder(embedding_size, input_size=input_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x.view(-1, *self.input_size))
        return self.decode(z)
