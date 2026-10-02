from src.data import load_data, validate_data


def test_wine_data_split_and_validation():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[1] == 13
    assert X_test.shape[1] == 13
    assert len(X_train) + len(X_test) == 178
    validate_data(X_train, y_train)
    validate_data(X_test, y_test)
