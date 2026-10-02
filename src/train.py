import argparse
import itertools

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_validate
from mlflow.models import infer_signature

from src.data import load_data

EXPERIMENT_NAME = "Wine-Cultivar-Classification"
MODEL_NAME = "WineClassifier"


def candidate_configs():
    grids = {
        "random_forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5],
        },
        "gradient_boosting": {
            "n_estimators": [100, 150, 200],
            "learning_rate": [0.05, 0.1],
        },
    }
    for family, grid in grids.items():
        keys = list(grid)
        for values in itertools.product(*(grid[key] for key in keys)):
            yield family, dict(zip(keys, values))


def make_model(family, params):
    if family == "random_forest":
        return RandomForestClassifier(random_state=42, **params)
    return GradientBoostingClassifier(random_state=42, **params)


def run_training(register=True):
    X_train, X_test, y_train, y_test = load_data()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "f1": "f1_macro",
        "accuracy": "accuracy",
        "neg_log_loss": "neg_log_loss",
    }
    best = None
    for family, params in candidate_configs():
        model = make_model(family, params)
        scores = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring,
            return_train_score=True
        )
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_f1 = float(np.mean(scores["test_f1"]))
        metrics = {
            "train_macro_f1": float(f1_score(y_train, train_pred, average="macro")),
            "validation_macro_f1": val_f1,
            "train_accuracy": float(accuracy_score(y_train, train_pred)),
            "validation_accuracy": float(np.mean(scores["test_accuracy"])),
            "train_log_loss": float(log_loss(y_train, model.predict_proba(X_train))),
            "validation_log_loss": float(-np.mean(scores["test_neg_log_loss"])),
        }
        with mlflow.start_run() as run:
            mlflow.log_params({"family": family, **params})
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"model_family": family, "evaluation": "5-fold stratified CV"})
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model, artifact_path="model", signature=signature,
                input_example=X_train.iloc[:5]
            )
            candidate = (val_f1, run.info.run_id)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if register and best is not None:
        client = mlflow.tracking.MlflowClient()
        model_uri = f"runs:/{best[1]}/model"
        registered = mlflow.register_model(model_uri, MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, "champion", registered.version)
    print(f"Best validation macro F1: {best[0]:.4f}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    run_training(register=not args.no_register)
