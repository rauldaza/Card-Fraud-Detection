"""
steps/train.py — SageMaker Processing Step: Training

SageMaker mounts:
  /opt/ml/processing/input/train/  → transactions_train.feather
  /opt/ml/processing/input/test/   → transactions_test.feather
  /opt/ml/processing/input/config/ → preprocessing_config.pkl
  /opt/ml/processing/output/       → trained checkpoint .pth + mlflow.db
"""

import os
import matplotlib.pyplot as plt

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import argparse
import logging
from dataclasses import dataclass, field

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.torch_lib.TabularTransformer import TabularTransformer, TransactionDataset
from utils.mlflow_utils import safe_log, setup_mlflow, upload_mlflow_db

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── SageMaker Paths ─────────────────────────────────────────────────────────
INPUT_TRAIN  = "/opt/ml/processing/input/train"
INPUT_TEST   = "/opt/ml/processing/input/test"
INPUT_CONFIG = "/opt/ml/processing/input/config"
OUTPUT_DIR   = "/opt/ml/processing/output"

TRAIN_PATH = os.path.join(INPUT_TRAIN,  "transactions_train.feather")
TEST_PATH  = os.path.join(INPUT_TEST,   "transactions_test.feather")
PKL_PATH   = os.path.join(INPUT_CONFIG, "preprocessing_config.pkl")
CKPT_OUT   = os.path.join(OUTPUT_DIR,   "TabularTransformer_checkpoint.pth")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Dataclass: Epoch Metrics ────────────────────────────────────────────────
# Accumulates per-epoch metrics during training, then hands them to MLflow
# in one clean batch call at the end. This avoids calling mlflow.log_metrics()
# inside the training loop, keeping the loop fast and the MLflow logic separate.

def name_normalization(columns):
    """Lowercase and replace hyphens with underscores."""
    if isinstance(columns, str):
        return columns.lower().replace('-', '_')
    return [c.lower().replace('-', '_') for c in columns]

@dataclass
class EpochRecord:
    epoch:          int
    train_loss:     float
    train_acc:      float
    test_loss:      float
    test_acc:       float


@dataclass
class TrainingHistory:
    records: list[EpochRecord] = field(default_factory=list)

    def append(self, record: EpochRecord):
        self.records.append(record)

    def log_to_mlflow(self):
        """
        Log all accumulated epoch metrics to MLflow in one pass.
        Using the `step` parameter tells MLflow which epoch each value belongs to,
        so you get proper time-series charts in the UI.
        """
        for r in self.records:
            safe_log(mlflow.log_metrics, {
                "train_loss": r.train_loss,
                "train_acc":  r.train_acc,
                "test_loss":  r.test_loss,
                "test_acc":   r.test_acc,
            }, step=r.epoch)

    def log_summary_to_mlflow(self):
        """
        Log best/final values as scalar summary metrics.
        These appear as single numbers in the MLflow run overview —
        useful for quick comparison across runs without opening charts.
        """
        if not self.records:
            return
        best = min(self.records, key=lambda r: r.test_loss)
        last = self.records[-1]
        safe_log(mlflow.log_metrics, {
            "best_test_loss":  best.test_loss,
            "best_test_acc":   best.test_acc,
            "best_epoch":      float(best.epoch),
            "final_test_loss": last.test_loss,
            "final_test_acc":  last.test_acc,
        })

    def print_summary(self):
        logger.info("=" * 50)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 50)
        for r in self.records:
            logger.info(
                f"  Epoch {r.epoch + 1:>2} | "
                f"Train Loss: {r.train_loss:.4f}  Acc: {100 * r.train_acc:.1f}% | "
                f"Test  Loss: {r.test_loss:.4f}  Acc: {100 * r.test_acc:.1f}%"
            )
        logger.info("=" * 50)

def save_loss_curve(history: TrainingHistory, output_dir: str) -> str:
    """Save a loss/accuracy curve plot and return its path."""
    epochs     = [r.epoch + 1 for r in history.records]
    train_loss = [r.train_loss for r in history.records]
    test_loss  = [r.test_loss  for r in history.records]
    train_acc  = [r.train_acc  for r in history.records]
    test_acc   = [r.test_acc   for r in history.records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, marker="o", label="Train Loss")
    ax1.plot(epochs, test_loss,  marker="o", label="Test Loss")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_acc, marker="o", label="Train Acc")
    ax2.plot(epochs, test_acc,  marker="o", label="Test Acc")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(path)
    plt.close()
    return path
# ─── Focal Loss ───────────────────────────────────────────────────────────────
# Standard CrossEntropy treats all mistakes equally.
# FocalLoss down-weights easy examples and focuses learning on hard ones —
# very useful for fraud detection where the vast majority of samples are non-fraud.

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha   = alpha
        self.gamma   = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce         = self.ce_loss(inputs, targets)
        pt         = torch.exp(-ce)
        focal      = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


# ─── Training & Evaluation ───────────────────────────────────────────────────
def run_epoch_train(model, loader, loss_fn, optimizer, device) -> tuple[float, float]:
    """One full pass over the training set. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct = 0.0, 0

    for batch_idx, (X_cat, X_cont, y) in enumerate(loader):
        X_cat, X_cont, y = X_cat.to(device), X_cont.to(device), y.to(device)

        pred = model(X_cat, X_cont)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        correct    += (pred.argmax(1) == y).float().sum().item()

        if batch_idx % 10 == 0:
            seen = batch_idx * loader.batch_size + len(X_cat)
            logger.info(f"  loss: {loss.item():.6f}  [{seen:>5d}/{len(loader.dataset):>5d}]")

    return total_loss / len(loader), correct / len(loader.dataset)


def run_epoch_eval(model, loader, loss_fn, device) -> tuple[float, float]:
    """One full pass over the evaluation set. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct = 0.0, 0

    with torch.no_grad():
        for X_cat, X_cont, y in loader:
            X_cat, X_cont, y = X_cat.to(device), X_cont.to(device), y.to(device)
            pred        = model(X_cat, X_cont)
            total_loss += loss_fn(pred, y).item()
            correct    += (pred.argmax(1) == y).float().sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # ── Args ─────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--batch-size",    type=int,   default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embed-dim",     type=int,   default=16)
    parser.add_argument("--target-col",    type=str,   default="isFraud")
    args = parser.parse_args()

    args.target_col = name_normalization(args.target_col)

    # ── Validate inputs ──────────────────────────────────────────────────────
    required = {"Train Data": TRAIN_PATH, "Test Data": TEST_PATH, "Preprocessing Config": PKL_PATH}
    missing  = [name for name, path in required.items() if not os.path.exists(path)]
    if missing:
        logger.error(f"Missing required files: {', '.join(missing)}")
        exit(1)
    for name, path in required.items():
        logger.info(f"  [FOUND] {name}: {path}")

    # ── Device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("GPU: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("GPU: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU — falling back to CPU")

    # ── Data ─────────────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    train_dataset = TransactionDataset(TRAIN_PATH, PKL_PATH, target_col=args.target_col)
    test_dataset  = TransactionDataset(TEST_PATH,  PKL_PATH, target_col=args.target_col)
    logger.info(f"  Train rows: {len(train_dataset):,}  |  Test rows: {len(test_dataset):,}")

    loader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    loader_test  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────────────
    n_categories = train_dataset.get_n_categories()
    n_continuous = train_dataset.num_idx[1]
    model = TabularTransformer(
        n_categories=n_categories,
        n_continuous=n_continuous,
        n_classes=2,
        embed_dim=args.embed_dim,
    ).to(device)

    loss_fn   = FocalLoss(alpha=1.0, gamma=2.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # ── MLflow setup ─────────────────────────────────────────────────────────
    mlflow_db = os.path.join(OUTPUT_DIR, "mlflow.db")
    setup_mlflow(mlflow_db)

    job_name = os.environ.get("SAGEMAKER_JOB_NAME", "local")

    # ── Training loop ────────────────────────────────────────────────────────
    history = TrainingHistory()

    with mlflow.start_run(run_name=f"training-{job_name}"):

        # Log hyperparameters once, at the start of the run
        safe_log(mlflow.log_params, {
            **vars(args),
            "instance_type": os.environ.get("SM_CURRENT_INSTANCE_TYPE", "unknown"),
            "device":        str(device),
        })

        logger.info(f"Starting training — {args.epochs} epoch(s) on {device}")
        logger.info("=" * 50)

        for epoch in range(args.epochs):
            logger.info(f"EPOCH {epoch + 1} / {args.epochs}")

            train_loss, train_acc = run_epoch_train(model, loader_train, loss_fn, optimizer, device)
            test_loss,  test_acc  = run_epoch_eval(model, loader_test,  loss_fn, device)

            record = EpochRecord(epoch, train_loss, train_acc, test_loss, test_acc)
            history.append(record)

            logger.info(
                f"  Train → Loss: {train_loss:.4f}  Acc: {100 * train_acc:.1f}% | "
                f"Test  → Loss: {test_loss:.4f}  Acc: {100 * test_acc:.1f}%"
            )

        # Log all epoch metrics to MLflow in one pass after training is done
        history.log_to_mlflow()
        history.log_summary_to_mlflow()
        loss_curve_path = save_loss_curve(history, OUTPUT_DIR)
        safe_log(mlflow.log_artifact, loss_curve_path, artifact_path="plots")
        history.print_summary()

        # ── Save checkpoint ──────────────────────────────────────────────────
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": {
                "n_categories": n_categories,
                "n_continuous":  n_continuous,
                "n_classes":     2,
                "embed_dim":     args.embed_dim,
            },
        }
        torch.save(checkpoint, CKPT_OUT)
        logger.info(f"Checkpoint saved → {CKPT_OUT}")

        # Log artifacts (model weights, preprocessing config, checkpoint file)
        safe_log(mlflow.pytorch.log_model, model,    artifact_path="model")
        safe_log(mlflow.log_artifact,      PKL_PATH, artifact_path="config")
        safe_log(mlflow.log_artifact,      CKPT_OUT, artifact_path="checkpoint")

    # Upload db AFTER start_run() context closes so the run is fully written to disk
    upload_mlflow_db(mlflow_db)
    logger.info("Done. SageMaker will upload the output directory to S3.")


if __name__ == "__main__":
    main()