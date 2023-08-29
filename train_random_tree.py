from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def load_preprocessed_dataset(dataset_name):
    """
    Loads the preprocessed dataset from the HuggingFace Hub.
    """
    data = load_dataset(dataset_name)
    train_embeddings = data["train"]["embedding"]
    train_labels = data["train"]["label"]
    test_embeddings = data["test"]["embedding"]
    test_labels = data["test"]["label"]

    return train_embeddings, train_labels, test_embeddings, test_labels


if __name__ == "__main__":
    train_embeddings, train_labels, test_embeddings, test_labels = load_preprocessed_dataset("SneakyInsect/MNIST-preprocessed")

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_embeddings, train_labels)
    # Predict the labels of the test data
    predictions = clf.predict(test_embeddings)
    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy of the Random Forest classifier with embedding size {32}: {accuracy*100:.2f}%")
