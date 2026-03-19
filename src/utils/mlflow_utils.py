"""
utils/mlflow_utils.py — Shared MLflow helpers for all pipeline steps.

FIX: Use S3 as the MLflow artifact store so plots/files survive the container.
     The tracking DB (mlflow.db) syncs to S3 for metrics/params.
     Artifacts go directly to s3://MLFLOW_S3_BUCKET/mlflow/artifacts/ via MLflow's
     built-in S3 artifact store — no manual tar/sync needed.
"""

import logging
import os
import boto3
from botocore.exceptions import ClientError
import mlflow

logger = logging.getLogger(__name__)

# ─── S3 Config ───────────────────────────────────────────────────────────────
MLFLOW_S3_BUCKET        = "mlops-models-821263772694"
MLFLOW_S3_KEY           = "mlflow/mlflow.db"
MLFLOW_S3_ARTIFACT_ROOT = f"s3://{MLFLOW_S3_BUCKET}/mlflow/artifacts"


# ─── S3 Sync ─────────────────────────────────────────────────────────────────

def download_mlflow_db(local_path: str,
                       bucket: str = MLFLOW_S3_BUCKET,
                       key:    str = MLFLOW_S3_KEY):
    """Pull the shared mlflow.db from S3 before starting a run."""
    try:
        boto3.client("s3").download_file(bucket, key, local_path)
        logger.info(f"MLflow db downloaded  ← s3://{bucket}/{key}")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "404":
            logger.info("No mlflow.db in S3 yet — starting fresh.")
        else:
            logger.warning(f"Could not download mlflow.db ({code}) — starting fresh.")


def upload_mlflow_db(local_path: str,
                     bucket: str = MLFLOW_S3_BUCKET,
                     key:    str = MLFLOW_S3_KEY):
    """
    Push the updated mlflow.db back to S3 after a run closes.
    Call OUTSIDE `with mlflow.start_run()` so the run is fully flushed first.
    Set LOCAL_MODE=1 to skip the upload during local testing.
    """
    if os.environ.get("LOCAL_MODE") == "1":
        logger.info("LOCAL_MODE=1 — skipping MLflow S3 upload.")
        return
    boto3.client("s3").upload_file(local_path, bucket, key)
    logger.info(f"MLflow db uploaded    → s3://{bucket}/{key}")


# ─── Safe Logging ────────────────────────────────────────────────────────────

def safe_log(func, *args, **kwargs):
    """Wrap any mlflow.log_* call so a failure never crashes the job."""
    try:
        func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"MLflow logging skipped (non-fatal): {e}")


# ─── Setup Helper ────────────────────────────────────────────────────────────

def setup_mlflow(local_db_path: str, experiment: str = "TabularTransformer"):
    """
    Download the shared DB, point MLflow at it, and set the artifact root to S3.

    KEY CHANGE vs the old version:
      - mlflow.set_tracking_uri()  → still SQLite for metrics/params (cheap, fast)
      - mlflow.set_experiment() now passes artifact_location → S3 for files
    
    This means mlflow.log_artifact / log_figure / log_table all write directly
    to S3 instead of the local container disk, so nothing is lost when the
    SageMaker container exits.
    """
    download_mlflow_db(local_db_path)
    mlflow.set_tracking_uri(f"sqlite:///{local_db_path}")

    # Create (or retrieve) the experiment with S3 as the artifact root.
    # If the experiment already exists with a different artifact location,
    # MLflow keeps the original location — so the first run wins.
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        client.create_experiment(experiment, artifact_location=MLFLOW_S3_ARTIFACT_ROOT)
        logger.info(f"Created experiment '{experiment}' with artifact root: {MLFLOW_S3_ARTIFACT_ROOT}")
    else:
        logger.info(f"Using existing experiment '{experiment}' (artifact root: {exp.artifact_location})")

    mlflow.set_experiment(experiment)
    logger.info(f"MLflow ready  |  experiment: {experiment}  |  db: {local_db_path}")
