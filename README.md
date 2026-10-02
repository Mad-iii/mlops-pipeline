# Wine MLOps Pipeline

Reproducible classification workflow for the sklearn Wine dataset.

## Setup

Requires Python 3.10.

```bash
make install
make lint
make test
```

## Training and evaluation

```bash
make train
python -m src.evaluate
```

Training performs five-fold stratified cross-validation for Random Forest and
Gradient Boosting configurations, logs parameters and train/validation metrics
to the `Wine-Cultivar-Classification` MLflow experiment, and registers the
highest validation macro-F1 run as `WineClassifier` with the `champion` alias.

The local tracking store is `sqlite:///mlflow.db`. To inspect runs and the
registry, start the UI with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open the local MLflow UI address shown in the terminal.

## Git workflow and report evidence

Use a feature branch for changes, open a pull request to `main`, and retain
your own commit history. For the required conflict exercise, create
`conflict-simulation`, change the same configuration line on that branch and
on `main`, merge, resolve the conflict markers manually, and commit the
resolution.

Capture your actual MLflow UI, passing GitHub Actions run, and Git history
screenshots for the report. These runtime and repository artifacts must be
produced from your own execution; they are not fabricated here.
