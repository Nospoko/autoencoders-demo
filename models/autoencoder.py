import torch
from torch import nn
import torch.utils.data
from torch.nn import functional as F

from models.layers import CNN_Decoder, CNN_Encoder


class Network(nn.Module):
    def __init__(self, cfg, input_size):
        super(Network, self).__init__()
        self.cfg = cfg
        output_size = cfg.model.embedding_size
        self.encoder = CNN_Encoder(output_size, input_size=input_size)
        self.decoder = CNN_Decoder(cfg.model.embedding_size, input_size=input_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        if self.cfg.dataset.name == "CIFAR10":
            z = self.encode(x.view(-1, 3 * 32 * 32))
        else:
            z = self.encode(x.view(-1, 28 * 28))
        return self.decode(z)


class Autoencoder(object):
    def __init__(self, cfg, input_size):
        self.cfg = cfg
        self.device = torch.device("cuda" if not cfg.system.no_cuda and torch.cuda.is_available() else "cpu")

        self.model = Network(cfg, input_size)
        self.model.to(self.device)

    def loss_function(self, recon_x, x):
        if self.cfg.dataset.name == "CIFAR10":
            BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 32 * 32), x.view(-1, 3 * 32 * 32), reduction="sum")
        else:
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28 * 28), reduction="sum")
        return BCE
