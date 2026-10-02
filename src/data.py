from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_data(test_size=0.2, random_state=42):
    dataset = load_wine(as_frame=True)
    X = dataset.data
    y = dataset.target
    validate_data(X, y)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def validate_data(X, y):
    if X.isnull().values.any() or y.isnull().any():
        raise ValueError("Dataset contains null values.")
    if X.shape[1] != 13:
        raise ValueError("Wine dataset must contain exactly 13 features.")
