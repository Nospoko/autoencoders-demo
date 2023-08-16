import torch

from data_loader import get_data_loaders
from train import initialize_model, visualize_ecg_reconstruction

# change the path to the checkpoint you want to test
checkpoint_path = "checkpoints/ECG_AE_ltafdb_checkpoint_epoch_10.pt"
checkpoint = torch.load(checkpoint_path)

train_loader, test_loader, input_size = get_data_loaders(checkpoint["config"])

autoencoder_instance = initialize_model(checkpoint["config"], train_loader, test_loader, input_size)
autoencoder_instance.model.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.Adam(autoencoder_instance.model.parameters(), checkpoint["config"].train.lr)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


visualize_ecg_reconstruction(checkpoint["config"], autoencoder_instance, test_loader)
