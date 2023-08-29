import torch

from train import initialize_model
from utils.data_loader import get_data_loaders
from utils.visualizations import draw_interpolation_grid, save_img_tensors_as_grid

checkpoint_path = "checkpoints/VAE_CIFAR10_checkpoint_epoch_1_embSize_64.pt"
checkpoint = torch.load(checkpoint_path)
train_loader, test_loader, input_size = get_data_loaders(checkpoint["config"], return_targets=True)
autoencoder_instance = initialize_model(checkpoint["config"], input_size)
autoencoder_instance.load_state_dict(checkpoint["model_state_dict"])


# visualize_embedding(checkpoint["config"], autoencoder_instance, test_loader)
save_img_tensors_as_grid(
    autoencoder_instance,
    test_loader,
    4,
    f"{checkpoint['config'].logger.results_path}side_by_side_comparison",
    side_by_side=True,
    type="VAE",
)
draw_interpolation_grid(checkpoint["config"], autoencoder_instance, test_loader)
