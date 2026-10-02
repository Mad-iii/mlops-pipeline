import time

from sklearn.metrics import f1_score
from src.data import load_data
from src.train import candidate_configs, make_model


def test_model_quality_gate():
    X_train, X_test, y_train, y_test = load_data()
    # Use the first configured candidate as a deterministic operational check.
    family, params = next(candidate_configs())
    model = make_model(family, params)
    model.fit(X_train, y_train)
    assert f1_score(y_test, model.predict(X_test), average="macro") >= 0.88

    start = time.perf_counter()
    predictions = model.predict(X_test)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms <= 30
    assert set(predictions).issubset({0, 1, 2})
