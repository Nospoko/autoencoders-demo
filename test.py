import torch
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from train import initialize_model
from utils.data_loader import get_data_loaders
from preprocess_dataset import create_embeddings
from utils.visualizations import visualize_embedding

# change the path to the checkpoint you want to test
checkpoint_path = "checkpoints/AE_MNIST_checkpoint_epoch_10_embSize_8.pt"
checkpoint = torch.load(checkpoint_path)

train_loader, test_loader, input_size = get_data_loaders(checkpoint["config"], return_targets=True)

autoencoder_instance = initialize_model(checkpoint["config"], input_size)
autoencoder_instance.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.Adam(autoencoder_instance.parameters(), checkpoint["config"].train.lr)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# train random forest classifier
train_embeddings, train_labels = create_embeddings(checkpoint["config"], autoencoder_instance, train_loader)
test_embeddings, test_labels = create_embeddings(checkpoint["config"], autoencoder_instance, test_loader)

visualize_embedding(checkpoint["config"], autoencoder_instance, test_loader, num_trio=3)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(train_embeddings, train_labels)

# test random forest classifier
preds = clf.predict(test_embeddings)
print(f"Accuracy: {accuracy_score(test_labels, preds)*100:.2f}%")
