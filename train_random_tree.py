from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def load_preprocessed_dataset(dataset_name):
    """
    Loads the preprocessed dataset from the HuggingFace Hub.
    """
    data = load_dataset(dataset_name)
    embedding_sizes = [32, 16, 8]
    train_embeddings = []
    train_labels = []
    test_embeddings = []
    test_labels = []
    for embedding_size in embedding_sizes:
        train_embeddings.append(data["train"][f"embedding{embedding_size}"])
        train_labels.append(data["train"]["label"])
        test_embeddings.append(data["test"][f"embedding{embedding_size}"])
        test_labels.append(data["test"]["label"])

    return train_embeddings, train_labels, test_embeddings, test_labels


if __name__ == "__main__":
    train_embeddings, train_labels, test_embeddings, test_labels = load_preprocessed_dataset("SneakyInsect/MNIST-preprocessed")

    for i in range(3):
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(train_embeddings[i], train_labels[i])
        # Predict the labels of the test data
        predictions = clf.predict(test_embeddings[i])
        # Calculate the accuracy of the predictions
        accuracy = accuracy_score(test_labels[i], predictions)
        print(f"Accuracy of the Random Forest classifier with embedding size {2**(5-i)}: {accuracy*100:.2f}%")
