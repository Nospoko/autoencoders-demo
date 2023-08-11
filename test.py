import torch

from train import initialize_model, draw_interpolation_grid

# change the path to the checkpoint you want to test
checkpoint_path = "checkpoints/AE_MNIST_checkpoint_epoch_3.pt"
checkpoint = torch.load(checkpoint_path)

autoencoder_instance = initialize_model(checkpoint["config"])
autoencoder_instance.model.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.Adam(autoencoder_instance.model.parameters(), lr=1e-3)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

draw_interpolation_grid(checkpoint["config"], autoencoder_instance)
