import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from utils.visualizations import interpolate_embeddings


def draw_ecg_reconstructions(
    model: nn.Module,
    signals: torch.Tensor,
) -> plt.Figure:
    reconstructions = model(signals).detach().cpu()
    signals = signals.cpu()
    n_samples = signals.shape[0]

    fig, axes = plt.subplots(ncols=2, nrows=n_samples, figsize=[10, 2 * n_samples])
    for it in range(n_samples):
        left_ax = axes[it][0]
        right_ax = axes[it][1]
        left_ax.plot(signals[it][0], label="signal")
        left_ax.plot(reconstructions[it][0], label="reconstruction")
        left_ax.legend()

        right_ax.plot(signals[it][1], label="signal")
        right_ax.plot(reconstructions[it][1], label="reconstruction")
        right_ax.legend()

    return fig


def draw_interpolation_tower(model: nn.Module, signals: torch.Tensor, num_interps: int = 10) -> plt.Figure:
    embeddings = model.encode(signals)

    # This only works for the first two signals, if there's more we ignore
    left = embeddings[0]
    right = embeddings[1]
    interpolated_embeddings = interpolate_embeddings(
        left=left,
        right=right,
        num_interps=num_interps,
    )

    decoded_interpolations = model.decode(interpolated_embeddings)
    interpolated_signals = [signals[0].unsqueeze(0), decoded_interpolations, signals[1].unsqueeze(0)]
    interpolated_signals = torch.cat(interpolated_signals)

    # Include target signals
    n_samples = num_interps + 2
    fig, axes = plt.subplots(
        ncols=2,
        nrows=n_samples,
        figsize=[10, 1 * n_samples],
        gridspec_kw={"hspace": 0.0},
    )

    for it in range(n_samples):
        signal = interpolated_signals[it].detach().cpu()
        left_ax = axes[it][0]
        left_ax.plot(signal[0])
        left_ax.set_xlim(0, 1000)
        left_ax.xaxis.set_visible(False)

        right_ax = axes[it][1]
        right_ax.plot(signal[1])
        right_ax.set_xlim(0, 1000)
        right_ax.xaxis.set_visible(False)
