import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from train import initialize_model
from utils.data_loader import get_data_loaders
from preprocess_dataset import create_embeddings

accuracies = []
for embedding_size in range(1, 33):
    checkpoint_path = f"checkpoints/multiple/VAE_MNIST_checkpoint_epoch_5_embSize_{embedding_size}.pt"
    checkpoint = torch.load(checkpoint_path)
    train_loader, test_loader, input_size = get_data_loaders(checkpoint["config"], return_targets=True)
    autoencoder_instance = initialize_model(checkpoint["config"], input_size)
    autoencoder_instance.load_state_dict(checkpoint["model_state_dict"])
    test_embeddings, test_labels = create_embeddings(checkpoint["config"], autoencoder_instance, test_loader)
    train_embeddings, train_labels = create_embeddings(checkpoint["config"], autoencoder_instance, train_loader)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_embeddings, train_labels)
    preds = clf.predict(test_embeddings)
    accuracies.append(accuracy_score(test_labels, preds) * 100)
    print(accuracies[-1])

plt.plot(range(1, 33), accuracies)
plt.xlabel("Embedding Size")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs. Embedding Size")
plt.show()
