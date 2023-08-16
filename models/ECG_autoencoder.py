import torch
from torch import nn

from models.layers import ECG_Decoder, ECG_Encoder


class Network(nn.Module):
    def __init__(self, cfg, input_size):
        super(Network, self).__init__()
        self.cfg = cfg
        self.input_size = input_size
        output_size = cfg.model.embedding_size
        self.encoder = ECG_Encoder(output_size, input_size=input_size)
        self.decoder = ECG_Decoder(cfg.model.embedding_size, input_size=input_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x.view(-1, *self.input_size))
        return self.decode(z)


class ECG_autoencoder(object):
    def __init__(self, cfg, train_loader, test_loader, input_size):
        self.cfg = cfg
        self.device = torch.device("cuda" if not cfg.system.no_cuda and torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.model = Network(cfg, input_size).to(self.device)
