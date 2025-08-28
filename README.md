# 📈 MLflow Tracking – Sample Project

Minimal, production-style MLflow example:
- Logs **params**, **metrics**, **artifacts** (confusion matrix), and the **model**
- Uses scikit-learn (Iris) with train/val split
- Includes a tiny **hyperparam tuner** to generate multiple runs
- Uses local tracking in `./mlruns` by default

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

### 🚀 Quick start

Train one model and log to MLflow:

```python3  src/train.py --model rf --n_estimators 200 --max_depth 5```

### 📊 View results

Launch MLflow UI (in another terminal):

```mlflow ui --backend-store-uri ./mlruns --port 5000```
Then open http://localhost:5000.

You can:

- Compare runs side by side
- Sort by accuracy or F1
- View confusion matrix plots under Artifacts
- Inspect saved models

### 📦 What gets logged
- Params: model type, n_estimators, max_depth, test_size, random_state
- Metrics: accuracy, macro_f1
- Artifacts: plots/confusion_matrix.png
- Model: sklearn pipeline logged with mlflow.sklearn.log_model
- All logs are stored under ./mlruns.

### 🧪 Tests

Pytest smoke tests check that:
- train.py runs end-to-end and produces an MLflow run
- tune.py produces multiple runs
| ⚠️ Make sure to add PYTHONPATH=src in front of pytest so imports resolve correctly.
Run tests:

```bash 
PYTHONPATH=src pytest -v
```
