import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from data_loader import get_data_loaders
from models.layers import CNN_Decoder, CNN_Encoder


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.args = args
        output_size = args.model.embedding_size
        if args.dataset.name == "CIFAR10":
            self.encoder = CNN_Encoder(output_size, input_size=(3, 32, 32))
            self.decoder = CNN_Decoder(args.model.embedding_size, input_size=(3, 32, 32))
        else:
            self.encoder = CNN_Encoder(output_size)
            self.decoder = CNN_Decoder(args.model.embedding_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        if self.args.dataset.name == "CIFAR10":
            z = self.encode(x.view(-1, 3 * 32 * 32))
        else:
            z = self.encode(x.view(-1, 28 * 28))
        return self.decode(z)


class Autoencoder(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if not args.system.no_cuda and torch.cuda.is_available() else "cpu")

        self.train_loader, self.test_loader = get_data_loaders(args)

        self.model = Network(args)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def loss_function(self, recon_x, x):
        if self.args.dataset.name == "CIFAR10":
            BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 32 * 32), x.view(-1, 3 * 32 * 32), reduction="sum")
        else:
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28 * 28), reduction="sum")
        return BCE
