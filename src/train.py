from __future__ import annotations
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from data import load_iris_split


def build_pipeline(model: str, n_estimators: int, max_depth: int) -> Pipeline:
    if model == "rf":
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth or None, random_state=42
        )
    elif model == "logreg":
        clf = LogisticRegression(max_iter=500)
    else:
        raise ValueError(f"Unknown model '{model}'")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def plot_confusion(y_true, y_pred, target_names, out_path: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=target_names, cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train and log to MLflow (Iris).")
    ap.add_argument("--experiment", type=str, default="iris-mlflow-demo")
    ap.add_argument("--tracking_uri", type=str, default=None, help="Override MLflow tracking URI")
    ap.add_argument("--model", type=str, default="rf", choices=["rf", "logreg"])
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--max_depth", type=int, default=5)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    # Configure MLflow (local ./mlruns if not provided)
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    else:
        mlflow.set_tracking_uri(str(Path("mlruns").resolve()))

    mlflow.set_experiment(args.experiment)

    # Prepare data & model
    data = load_iris_split(test_size=args.test_size, random_state=args.random_state)
    pipe = build_pipeline(args.model, args.n_estimators, args.max_depth)

    with mlflow.start_run(run_name=f"{args.model}-d{args.max_depth}-n{args.n_estimators}") as run:
        # Log parameters
        mlflow.log_param("model", args.model)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        # Fit & evaluate
        pipe.fit(data.X_train, data.y_train)
        pred = pipe.predict(data.X_val)

        acc = accuracy_score(data.y_val, pred)
        f1 = f1_score(data.y_val, pred, average="macro")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("macro_f1", f1)

        # Artifacts: confusion matrix
        cm_path = Path("artifacts/confusion_matrix.png")
        plot_confusion(data.y_val, pred, data.target_names, cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # Log the model
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        print(f"Run id: {run.info.run_id}")
        print(f"accuracy={acc:.4f} macro_f1={f1:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
