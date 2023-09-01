import torch
import torch.nn as nn
from matplotlib import pyplot as plt


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
