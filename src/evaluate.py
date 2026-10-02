import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import load_data
from src.train import MODEL_NAME


def evaluate_champion():
    _, X_test, _, y_test = load_data()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@champion"
    )

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    metrics = {
        "test_macro_f1": f1_score(
            y_test, predictions, average="macro"
        ),
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_log_loss": log_loss(y_test, probabilities),
    }

    print("Champion Model Test Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics


if __name__ == "__main__":
    evaluate_champion()
