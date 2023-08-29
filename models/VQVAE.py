import torch
from torch import nn
from torch.nn import functional as F


class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        # Section 4.1 of the paper
        self.layers = nn.ModuleList()

        for _ in range(num_residual_layers):
            self.layers.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(num_hiddens, num_residual_hiddens, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(num_residual_hiddens, num_hiddens, kernel_size=1),
                )
            )

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = h + layer(h)

        return F.relu(h)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_downsampling_layers, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        # Initializations
        self.downsampling_layers = nn.ModuleList()
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)

        # Layer Definitions
        for i in range(num_downsampling_layers):
            if i == 0:
                out_channels = num_hiddens // 2
            elif i == 1:
                in_channels, out_channels = num_hiddens // 2, num_hiddens
            else:
                in_channels, out_channels = num_hiddens, num_hiddens

            self.downsampling_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1), nn.ReLU()
                )
            )

        self.final_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, padding=1)

    def forward(self, x):
        # Forward Pass
        for layer in self.downsampling_layers:
            x = layer(x)

        x = self.final_conv(x)
        return self.residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_hiddens, num_upsampling_layers, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        # Initializations
        self.upsampling_layers = nn.ModuleList()
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)

        # Layer Definitions
        self.initial_conv = nn.Conv2d(in_channels=embedding_dim, out_channels=num_hiddens, kernel_size=3, padding=1)

        for i in range(num_upsampling_layers):
            if i < num_upsampling_layers - 2:
                in_channels, out_channels = num_hiddens, num_hiddens
            elif i == num_upsampling_layers - 2:
                in_channels, out_channels = num_hiddens, num_hiddens // 2
            else:
                in_channels, out_channels = num_hiddens // 2, 3

            self.upsampling_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
                    nn.ReLU() if i < num_upsampling_layers - 1 else nn.Identity(),
                )
            )

    def forward(self, x):
        # Forward Pass
        x = self.initial_conv(x)
        x = self.residual_stack(x)

        for layer in self.upsampling_layers:
            x = layer(x)

        return x


class VQVAE(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()

        in_channels = cfg.in_channels
        num_hiddens = cfg.num_hiddens
        num_downsampling_layers = cfg.num_downsampling_layers
        num_residual_layers = cfg.num_residual_layers
        num_residual_hiddens = cfg.num_residual_hiddens
        embedding_dim = cfg.embedding_dim
        num_embeddings = cfg.num_embeddings
        use_ema = cfg.use_ema
        decay = cfg.decay
        epsilon = cfg.epsilon

        self.device = torch.device("cuda" if device and torch.cuda.is_available() else "cpu")
        print(self.device)

        # Define layers in __init__
        self.encoder_layer = self.build_encoder(
            in_channels,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
        )
        self.pre_vq_conv_layer = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=embedding_dim,
            kernel_size=1,
        )
        self.vector_quantizer = self.build_vector_quantizer(
            embedding_dim,
            num_embeddings,
            use_ema,
            decay,
            epsilon,
        )
        self.decoder_layer = self.build_decoder(
            embedding_dim,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
        )

    def build_encoder(self, in_channels, num_hiddens, num_downsampling_layers, num_residual_layers, num_residual_hiddens):
        return Encoder(
            in_channels,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
        )

    def build_vector_quantizer(self, embedding_dim, num_embeddings, use_ema, decay, epsilon):
        return VectorQuantizer(embedding_dim, num_embeddings, use_ema, decay, epsilon)

    def build_decoder(self, embedding_dim, num_hiddens, num_downsampling_layers, num_residual_layers, num_residual_hiddens):
        return Decoder(embedding_dim, num_hiddens, num_downsampling_layers, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        # Use the layers in forward
        encoded_x = self.encoder_layer(x)

        pre_vq_x = self.pre_vq_conv_layer(encoded_x)
        (
            quantized_x,
            dictionary_loss,
            commitment_loss,
            encoding_indices,
        ) = self.vector_quantizer(pre_vq_x)

        x_recon = self.decoder_layer(quantized_x)

        return {
            "dictionary_loss": dictionary_loss,
            "commitment_loss": commitment_loss,
            "x_recon": x_recon,
        }


class SonnetExponentialMovingAverage(nn.Module):
    # See: https://github.com/deepmind/sonnet/blob/5cbfdc356962d9b6198d5b63f0826a80acfdf35b/sonnet/src/moving_averages.py#L25.
    # They do *not* use the exponential moving average updates described in Appendix A.1
    # of "Neural Discrete Representation Learning".
    def __init__(self, decay, shape):
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros(*shape))
        self.register_buffer("average", torch.zeros(*shape))

    def update(self, value):
        self.counter += 1
        with torch.no_grad():
            self.hidden -= (self.hidden - value) * (1 - self.decay)
            self.average = self.hidden / (1 - self.decay**self.counter)

    def __call__(self, value):
        self.update(value)
        return self.average


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, use_ema, decay, epsilon):
        super().__init__()
        # See Section 3 of "Neural Discrete Representation Learning" and:
        # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L142.

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.use_ema = use_ema
        # Weight for the exponential moving average.
        self.decay = decay
        # Small constant to avoid numerical instability in embedding updates.
        self.epsilon = epsilon

        # Dictionary embeddings.
        limit = 3**0.5
        e_i_ts = torch.FloatTensor(embedding_dim, num_embeddings).uniform_(-limit, limit)
        if use_ema:
            self.register_buffer("e_i_ts", e_i_ts)
        else:
            self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))

        # Exponential moving average of the cluster counts.
        self.N_i_ts = SonnetExponentialMovingAverage(decay, (num_embeddings,))
        # Exponential moving average of the embeddings.
        self.m_i_ts = SonnetExponentialMovingAverage(decay, e_i_ts.shape)

    def forward(self, x):
        flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        distances = (flat_x**2).sum(1, keepdim=True) - 2 * flat_x @ self.e_i_ts + (self.e_i_ts**2).sum(0, keepdim=True)
        encoding_indices = distances.argmin(1)
        quantized_x = F.embedding(encoding_indices.view(x.shape[0], *x.shape[2:]), self.e_i_ts.transpose(0, 1)).permute(
            0, 3, 1, 2
        )

        # See second term of Equation (3).
        if not self.use_ema:
            dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()
        else:
            dictionary_loss = None

        # See third term of Equation (3).
        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()
        # Straight-through gradient. See Section 3.2.
        quantized_x = x + (quantized_x - x).detach()

        if self.use_ema and self.training:
            with torch.no_grad():
                # See Appendix A.1 of "Neural Discrete Representation Learning".

                # Cluster counts.
                encoding_one_hots = F.one_hot(encoding_indices, self.num_embeddings).type(flat_x.dtype)
                n_i_ts = encoding_one_hots.sum(0)
                # Updated exponential moving average of the cluster counts.
                # See Equation (6).
                self.N_i_ts(n_i_ts)

                # Exponential moving average of the embeddings. See Equation (7).
                embed_sums = flat_x.transpose(0, 1) @ encoding_one_hots
                self.m_i_ts(embed_sums)

                # This is kind of weird. <- comment from the original code.
                # Compare: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L270
                # and Equation (8).
                N_i_ts_sum = self.N_i_ts.average.sum()
                N_i_ts_stable = (
                    (self.N_i_ts.average + self.epsilon) / (N_i_ts_sum + self.num_embeddings * self.epsilon) * N_i_ts_sum
                )
                self.e_i_ts = self.m_i_ts.average / N_i_ts_stable.unsqueeze(0)

        return (
            quantized_x,
            dictionary_loss,
            commitment_loss,
            encoding_indices.view(x.shape[0], -1),
        )
