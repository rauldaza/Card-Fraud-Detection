"""
tests/test_mlflow_utils.py — Unit tests for utils/mlflow_utils.py
=================================================================

All external calls (boto3, mlflow) are mocked so these tests run with
no AWS credentials and no live MLflow server.

Real usage from train.py / test.py:
    setup_mlflow(mlflow_db)           # download db, set tracking URI & experiment
    with mlflow.start_run():
        safe_log(mlflow.log_metrics, {...}, step=epoch)
        safe_log(mlflow.log_artifact, path, artifact_path="plots")
    upload_mlflow_db(mlflow_db)       # push updated db back to S3
"""

import logging
from unittest.mock import MagicMock, call, patch

import pytest
from botocore.exceptions import ClientError

import utils.mlflow_utils as mu


# ── Helpers ───────────────────────────────────────────────────────────────────

def _client_error(code: str) -> ClientError:
    """Build a ClientError with a given HTTP error code."""
    return ClientError({"Error": {"Code": code, "Message": "test"}}, "op")


# ── download_mlflow_db ────────────────────────────────────────────────────────

class TestDownloadMlflowDb:
    """download_mlflow_db pulls the shared mlflow.db from S3 before a run."""

    @patch("utils.mlflow_utils.boto3.client")
    def test_success_calls_download_file(self, mock_boto3_client):
        """Happy path: boto3.download_file is called with the right arguments."""
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        mu.download_mlflow_db("/tmp/mlflow.db", bucket="my-bucket", key="mlflow/mlflow.db")

        mock_s3.download_file.assert_called_once_with(
            "my-bucket", "mlflow/mlflow.db", "/tmp/mlflow.db"
        )

    @patch("utils.mlflow_utils.boto3.client")
    def test_404_is_silently_swallowed(self, mock_boto3_client, caplog):
        """
        A 404 means No db in S3 yet — starting fresh.
        The function must NOT raise, it should just log an info message.
        Mirrors the cold-start case on the first SageMaker training job.
        """
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = _client_error("404")
        mock_boto3_client.return_value = mock_s3

        with caplog.at_level(logging.INFO, logger="utils.mlflow_utils"):
            mu.download_mlflow_db("/tmp/mlflow.db")  # must not raise

        assert any("fresh" in msg for msg in caplog.messages)

    @patch("utils.mlflow_utils.boto3.client")
    def test_non_404_error_warns_and_does_not_raise(self, mock_boto3_client, caplog):
        """
        A non-404 error (e.g. 403 Forbidden) should log a WARNING but still
        not crash — the step continues with a fresh db.
        """
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = _client_error("403")
        mock_boto3_client.return_value = mock_s3

        with caplog.at_level(logging.WARNING, logger="utils.mlflow_utils"):
            mu.download_mlflow_db("/tmp/mlflow.db")  # must not raise

        assert any("403" in msg for msg in caplog.messages)


# ── upload_mlflow_db ──────────────────────────────────────────────────────────

class TestUploadMlflowDb:
    """upload_mlflow_db pushes the updated db back to S3 after the run closes."""

    @patch("utils.mlflow_utils.boto3.client")
    def test_calls_upload_file_with_correct_args(self, mock_boto3_client):
        """
        train.py calls upload_mlflow_db(mlflow_db) OUTSIDE start_run() context
        so the run is fully flushed first.  We verify the S3 target is correct.
        """
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        mu.upload_mlflow_db("/tmp/mlflow.db", bucket="my-bucket", key="mlflow/mlflow.db")

        mock_s3.upload_file.assert_called_once_with(
            "/tmp/mlflow.db", "my-bucket", "mlflow/mlflow.db"
        )

    @patch("utils.mlflow_utils.boto3.client")
    def test_uses_default_bucket_and_key(self, mock_boto3_client):
        """Default bucket/key constants mirror the real S3 layout."""
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        mu.upload_mlflow_db("/tmp/mlflow.db")

        mock_s3.upload_file.assert_called_once_with(
            "/tmp/mlflow.db", mu.MLFLOW_S3_BUCKET, mu.MLFLOW_S3_KEY
        )


# ── safe_log ──────────────────────────────────────────────────────────────────

class TestSafeLog:
    """
    safe_log wraps any mlflow.log_* call so a failure never crashes the job.
    Used extensively in train.py:
        safe_log(mlflow.log_metrics, {"train_loss": 0.4, ...}, step=epoch)
        safe_log(mlflow.log_artifact, loss_curve_path, artifact_path="plots")
    """

    def test_passes_args_and_kwargs_to_func(self):
        mock_fn = MagicMock()
        mu.safe_log(mock_fn, {"train_loss": 0.4}, step=1)
        mock_fn.assert_called_once_with({"train_loss": 0.4}, step=1)

    def test_returns_none_on_success(self):
        mock_fn = MagicMock(return_value="ignored")
        result = mu.safe_log(mock_fn)
        assert result is None

    def test_exception_is_swallowed_not_raised(self, caplog):
        """Any exception inside func is caught; the job must not crash."""
        def boom(*a, **kw):
            raise RuntimeError("MLflow server unreachable")

        with caplog.at_level(logging.WARNING, logger="utils.mlflow_utils"):
            mu.safe_log(boom, "some_arg")  # must not raise

        assert any("MLflow logging skipped" in msg for msg in caplog.messages)

    def test_exception_message_is_logged(self, caplog):
        """The original exception message should appear in the warning."""
        def boom(*a, **kw):
            raise ValueError("bad metric shape")

        with caplog.at_level(logging.WARNING, logger="utils.mlflow_utils"):
            mu.safe_log(boom)

        assert any("bad metric shape" in msg for msg in caplog.messages)


# ── setup_mlflow ───────────────────────────────────────────────────────────────

class TestSetupMlflow:
    """
    setup_mlflow orchestrates:
      1. download_mlflow_db  (pulls shared db from S3)
      2. mlflow.set_tracking_uri (SQLite)
      3. create or retrieve experiment (with S3 artifact root)
      4. mlflow.set_experiment
    """

    @patch("utils.mlflow_utils.mlflow.set_experiment")
    @patch("utils.mlflow_utils.mlflow.set_tracking_uri")
    @patch("utils.mlflow_utils.mlflow.tracking.MlflowClient")
    @patch("utils.mlflow_utils.download_mlflow_db")
    def test_creates_experiment_when_it_does_not_exist(
        self, mock_download, mock_client_cls, mock_set_uri, mock_set_exp
    ):
        """
        First run: experiment doesn't exist yet → client.create_experiment
        is called with the S3 artifact root.
        """
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None
        mock_client_cls.return_value = mock_client

        mu.setup_mlflow("/tmp/mlflow.db", experiment="TabularTransformer")

        mock_download.assert_called_once_with("/tmp/mlflow.db")
        mock_set_uri.assert_called_once_with("sqlite:////tmp/mlflow.db")
        mock_client.create_experiment.assert_called_once_with(
            "TabularTransformer", artifact_location=mu.MLFLOW_S3_ARTIFACT_ROOT
        )
        mock_set_exp.assert_called_once_with("TabularTransformer")

    @patch("utils.mlflow_utils.mlflow.set_experiment")
    @patch("utils.mlflow_utils.mlflow.set_tracking_uri")
    @patch("utils.mlflow_utils.mlflow.tracking.MlflowClient")
    @patch("utils.mlflow_utils.download_mlflow_db")
    def test_skips_creation_when_experiment_already_exists(
        self, mock_download, mock_client_cls, mock_set_uri, mock_set_exp
    ):
        """
        Subsequent runs: experiment already exists → create_experiment is NOT called.
        Mirrors what happens on every SageMaker job after the first.
        """
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = MagicMock(
            artifact_location="s3://existing-bucket/mlflow/artifacts"
        )
        mock_client_cls.return_value = mock_client

        mu.setup_mlflow("/tmp/mlflow.db", experiment="TabularTransformer")

        mock_client.create_experiment.assert_not_called()
        mock_set_exp.assert_called_once_with("TabularTransformer")
