import torch
import numpy as np
from torch import nn
from torch.autograd import Variable

# from torch.nn import functional as F


class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(1, 28, 28)):
        super(CNN_Encoder, self).__init__()

        self.input_size = input_size
        self.channel_mult = 16

        if self.input_size[1] == 32:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=self.channel_mult * 1, kernel_size=4, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.channel_mult * 1, self.channel_mult * 2, 4, 2, 1),
                nn.BatchNorm2d(self.channel_mult * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.channel_mult * 2, self.channel_mult * 4, 4, 2, 1),
                nn.BatchNorm2d(self.channel_mult * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.channel_mult * 4, self.channel_mult * 8, 4, 2, 1),
                nn.BatchNorm2d(self.channel_mult * 8),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=self.channel_mult * 1, kernel_size=4, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.channel_mult * 1, self.channel_mult * 2, 4, 2, 1),
                nn.BatchNorm2d(self.channel_mult * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.channel_mult * 2, self.channel_mult * 4, 4, 2, 1),
                nn.BatchNorm2d(self.channel_mult * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.channel_mult * 4, self.channel_mult * 8, 4, 2, 1),
                nn.BatchNorm2d(self.channel_mult * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.channel_mult * 8, self.channel_mult * 16, 3, 2, 1),
                nn.BatchNorm2d(self.channel_mult * 16),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.flat_fts = self.get_flat_fts(self.conv)

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2),
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, *self.input_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.conv(x.view(-1, *self.input_size))
        x = x.contiguous().view(-1, self.flat_fts)
        return self.linear(x)


class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size=(1, 28, 28)):
        super(CNN_Decoder, self).__init__()
        self.input_size = input_size

        self.input_height = self.input_size[1]
        self.input_width = self.input_size[2]
        self.input_dim = embedding_size
        self.channel_mult = 16
        self.output_channels = 3 if self.input_size[1] == 32 else 1
        self.fc_output_dim = 512

        self.fc = nn.Sequential(nn.Linear(self.input_dim, self.fc_output_dim), nn.BatchNorm1d(self.fc_output_dim), nn.ReLU(True))

        if self.input_size[1] == 32:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(self.fc_output_dim, self.channel_mult * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.channel_mult * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(self.channel_mult * 8, self.channel_mult * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.channel_mult * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(self.channel_mult * 4, self.channel_mult * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.channel_mult * 2),
                nn.ReLU(True),
                # Adding another upsampling layer to go from 16x16 to 32x32
                nn.ConvTranspose2d(self.channel_mult * 2, self.channel_mult * 1, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.channel_mult * 1),
                nn.ReLU(True),
                # Output layer
                nn.ConvTranspose2d(self.channel_mult * 1, self.output_channels, 4, 2, 1, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.deconv = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(self.fc_output_dim, self.channel_mult * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(self.channel_mult * 4),
                nn.ReLU(True),
                # state size. self.channel_mult*32 x 4 x 4
                nn.ConvTranspose2d(self.channel_mult * 4, self.channel_mult * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(self.channel_mult * 2),
                nn.ReLU(True),
                # state size. self.channel_mult*16 x 7 x 7
                nn.ConvTranspose2d(self.channel_mult * 2, self.channel_mult * 1, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.channel_mult * 1),
                nn.ReLU(True),
                # state size. self.channel_mult*8 x 14 x 14
                nn.ConvTranspose2d(self.channel_mult * 1, self.output_channels, 4, 2, 1, bias=False),
                nn.Sigmoid()
                # state size. self.output_channels x 28 x 28
            )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.fc_output_dim, 1, 1)
        x = self.deconv(x)
        return x  # We removed the flattening view at the end


class ECG_Encoder(nn.Module):
    def __init__(self, output_size, input_size):
        super(ECG_Encoder, self).__init__()

        # Assuming input_size = (channels, sequence_length)
        self.channels, self.seq_length = input_size

        self.layers = nn.Sequential(
            nn.Conv1d(self.channels, 16, kernel_size=3, stride=2, padding=1),  # [batch, 16, seq_length/2]
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),  # [batch, 32, seq_length/4]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # [batch, 64, seq_length/8]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * self.seq_length // 8, output_size),
        )

    def forward(self, x):
        return self.layers(x)


class ECG_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size):
        super(ECG_Decoder, self).__init__()

        self.channels, self.seq_length = input_size

        self.layers = nn.Sequential(
            nn.Linear(embedding_size, 64 * self.seq_length // 8),
            nn.ReLU(),
            nn.Unflatten(1, (64, self.seq_length // 8)),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, self.channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),  # Adjust this activation based on the nature of your input data
        )

    def forward(self, z):
        return self.layers(z)
