"""
steps/ft_test.py — SageMaker Processing Step: FTTransformer Evaluation

SageMaker mounts:
  /opt/ml/processing/input/test/   → transactions_test.feather
  /opt/ml/processing/input/model/  → FTTransformer_checkpoint.pth
  /opt/ml/processing/input/config/ → preprocessing_config.pkl
  /opt/ml/processing/output/       → plots + mlflow.db
"""

import os

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import argparse
import logging
import pickle
from dataclasses import dataclass

import torch
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    roc_curve,
    auc,
    balanced_accuracy_score,
)

from utils.torch_lib.FeatureTransformer import FTTransformer, FTTransformerWrapper
from utils.mlflow_utils import safe_log, setup_mlflow, upload_mlflow_db

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def name_normalization(columns):
    """Lowercase and replace hyphens with underscores."""
    if isinstance(columns, str):
        return columns.lower().replace('-', '_')
    return [c.lower().replace('-', '_') for c in columns]


# ─── Dataclass: Evaluation Results ───────────────────────────────────────────
@dataclass
class EvalResults:
    # Scalar metrics
    accuracy:          float
    f1:                float
    f1_weighted:       float
    balanced_accuracy: float
    roc_auc:           float
    # Curve data for plotting
    fpr:     "np.ndarray"
    tpr:     "np.ndarray"
    cm:      "np.ndarray"
    # Diagnostic table
    diag_df: pd.DataFrame

    def as_mlflow_dict(self) -> dict:
        """Return only the scalar metrics — the shape MLflow expects."""
        return {
            "test_accuracy":          self.accuracy,
            "test_f1":                self.f1,
            "test_f1_weighted":       self.f1_weighted,
            "test_balanced_accuracy": self.balanced_accuracy,
            "test_roc_auc":           self.roc_auc,
        }

    def print_summary(self):
        logger.info("=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"  Accuracy          : {self.accuracy:.4f}")
        logger.info(f"  F1                : {self.f1:.4f}")
        logger.info(f"  F1 Weighted       : {self.f1_weighted:.4f}")
        logger.info(f"  Balanced Accuracy : {self.balanced_accuracy:.4f}")
        logger.info(f"  ROC AUC           : {self.roc_auc:.4f}")
        logger.info("=" * 50)


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame, y_pred, y_proba, target_col: str) -> EvalResults:
    """Compute all metrics and return them in a structured EvalResults object."""
    y_test = df[target_col].values

    logger.info("=== CLASSIFICATION REPORT ===")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    diag_df = pd.DataFrame({
        "sample_id": df.index,
        "y_true":    y_test,
        "y_pred":    y_pred,
        "y_proba":   y_proba,
    })

    return EvalResults(
        accuracy          = accuracy_score(y_test, y_pred),
        f1                = f1_score(y_test, y_pred),
        f1_weighted       = f1_score(y_test, y_pred, average="weighted"),
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred),
        roc_auc           = auc(fpr, tpr),
        fpr               = fpr,
        tpr               = tpr,
        cm                = confusion_matrix(y_test, y_pred),
        diag_df           = diag_df,
    )


# ─── Plots ───────────────────────────────────────────────────────────────────

def save_confusion_matrix(results: EvalResults, path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    sns.heatmap(results.cm, annot=True, fmt="d",    cmap="Blues",  ax=ax1)
    ax1.set_title("Confusion Matrix (Counts)")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    cm_norm = results.cm.astype(float) / results.cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens", ax=ax2)
    ax2.set_title("Confusion Matrix (Normalized)")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_roc_curve(results: EvalResults, path: str):
    plt.figure(figsize=(8, 6))
    plt.plot(results.fpr, results.tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {results.roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Args ─────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data",  type=str, required=True)
    parser.add_argument("--model-dir",  type=str, required=True)
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--target-col", type=str, default="isFraud")
    args = parser.parse_args()

    args.target_col = name_normalization(args.target_col)

    # ── Output dirs ──────────────────────────────────────────────────────────
    plots_dir = os.path.join(args.output_dir, "plots")
    diag_dir  = os.path.join(args.output_dir, "diagnostics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(diag_dir,  exist_ok=True)

    mlflow_db = os.path.join(args.output_dir, "mlflow.db")
    setup_mlflow(mlflow_db)

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info(f"Loading test data: {args.test_data}")
    df     = pd.read_feather(args.test_data)
    X_test = df.drop(columns=[args.target_col])
    logger.info(f"  Test rows: {len(df):,}")

    # ── Load preprocessing config ─────────────────────────────────────────────
    pkl_path = os.path.join(args.config_dir, "preprocessing_config.pkl")
    logger.info(f"Loading preprocessing config: {pkl_path}")
    with open(pkl_path, "rb") as f:
        pkl_config = pickle.load(f)

    preprocessor = pkl_config["preprocessor"]
    num_idx      = pkl_config["num"]
    cat_idx      = pkl_config["cat"]
    n_num_cols   = len(pkl_config["num_cols"])

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.model_dir, "FTTransformer_checkpoint.pth")
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = FTTransformer(**ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])

    # ── Inference ────────────────────────────────────────────────────────────
    logger.info("Preprocessing and running inference...")
    X_transformed = preprocessor.transform(X_test)

    wrapped = FTTransformerWrapper(
        model      = model,
        num_idx    = num_idx,
        cat_idx    = cat_idx,
        n_num_cols = n_num_cols,
        batch_size = 128,
    )
    y_pred  = wrapped.predict(X_transformed)
    y_proba = wrapped.predict_proba(X_transformed)[:, 1]   # P(fraud)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    results = evaluate(df, y_pred, y_proba, args.target_col)
    results.print_summary()

    # ── Save artifacts ───────────────────────────────────────────────────────
    parquet_path  = os.path.join(diag_dir,  "eval_results.parquet")
    cm_plot_path  = os.path.join(plots_dir, "confusion_matrix.png")
    roc_plot_path = os.path.join(plots_dir, "roc_curve.png")

    results.diag_df.to_parquet(parquet_path)
    save_confusion_matrix(results, cm_plot_path)
    save_roc_curve(results, roc_plot_path)
    logger.info("Plots and diagnostics saved.")

    # ── MLflow logging ───────────────────────────────────────────────────────
    job_name = os.environ.get("SAGEMAKER_JOB_NAME", "local")

    with mlflow.start_run(run_name=f"ft-evaluation-{job_name}"):
        safe_log(mlflow.log_metrics,  results.as_mlflow_dict())
        safe_log(mlflow.log_table,    data=results.diag_df.head(1000), artifact_file="diagnostics/error_analysis.json")
        safe_log(mlflow.log_artifact, parquet_path,  artifact_path="diagnostics")
        safe_log(mlflow.log_artifact, cm_plot_path,  artifact_path="plots")
        safe_log(mlflow.log_artifact, roc_plot_path, artifact_path="plots")
        logger.info("Metrics, diagnostics, and plots logged to MLflow.")

    upload_mlflow_db(mlflow_db)


if __name__ == "__main__":
    main()
