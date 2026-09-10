from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def generate_dataset(samples=1000, features=20):
    X, y = make_classification(
        n_samples=samples,
        n_features=features,
        n_informative=10,
        n_classes=2,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)
